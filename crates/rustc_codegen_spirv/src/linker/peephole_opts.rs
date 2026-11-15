use super::id;
use rspirv::dr::{Function, Instruction, Module, ModuleHeader, Operand};
use rspirv::spirv::{Op, Word};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_middle::bug;

pub fn collect_types(module: &Module) -> FxHashMap<Word, Instruction> {
    module
        .types_global_values
        .iter()
        .filter_map(|inst| Some((inst.result_id?, inst.clone())))
        .collect()
}

fn composite_count(types: &FxHashMap<Word, Instruction>, ty_id: Word) -> Option<usize> {
    let ty = types.get(&ty_id)?;
    match ty.class.opcode {
        Op::TypeStruct => Some(ty.operands.len()),
        Op::TypeVector => Some(ty.operands[1].unwrap_literal_bit32() as usize),
        Op::TypeArray => {
            let length_id = ty.operands[1].unwrap_id_ref();
            let const_inst = types.get(&length_id)?;
            if const_inst.class.opcode != Op::Constant {
                return None;
            }
            let const_ty = types.get(&const_inst.result_type.unwrap())?;
            if const_ty.class.opcode != Op::TypeInt {
                return None;
            }
            let const_value = match const_inst.operands[0] {
                Operand::LiteralBit32(v) => v as usize,
                Operand::LiteralBit64(v) => v as usize,
                _ => bug!(),
            };
            Some(const_value)
        }
        _ => None,
    }
}

/// Given a chain of `OpCompositeInsert` instructions where all slots of the composite are
/// assigned, replace the chain with a single `OpCompositeConstruct`.
pub fn composite_construct(types: &FxHashMap<Word, Instruction>, function: &mut Function) {
    let defs = function
        .all_inst_iter()
        .filter_map(|inst| Some((inst.result_id?, inst.clone())))
        .collect::<FxHashMap<Word, Instruction>>();
    for block in &mut function.blocks {
        for inst in &mut block.instructions {
            if inst.class.opcode != Op::CompositeInsert {
                continue;
            }
            // Get the number of components to expect
            let component_count = match composite_count(types, inst.result_type.unwrap()) {
                Some(c) => c,
                None => continue,
            };
            // Remember a map of index -> value for that index. If any index is missing (None)
            // afterwards, then we know not all slots have been filled in, and we should skip
            // optimizing this chain.
            let mut components = vec![None; component_count];
            let mut cur_inst: &Instruction = inst;
            // Start looping from the current instruction, through each instruction in the chain.
            while cur_inst.class.opcode == Op::CompositeInsert {
                if cur_inst.operands.len() != 3 {
                    // If there's more than one index, skip optimizing this chain.
                    break;
                }
                let value = cur_inst.operands[0].unwrap_id_ref();
                let index = cur_inst.operands[2].unwrap_literal_bit32() as usize;
                if index >= components.len() {
                    // Theoretically shouldn't happen, as it's invalid SPIR-V if the index is out
                    // of bounds, but just stop optimizing instead of panicking here.
                    break;
                }
                if components[index].is_none() {
                    components[index] = Some(value);
                }
                // Follow back one in the chain of OpCompositeInsert
                cur_inst = match defs.get(&cur_inst.operands[1].unwrap_id_ref()) {
                    Some(i) => i,
                    None => break,
                };
            }
            // If all components are filled in (collect() returns Some), replace it with
            // `OpCompositeConstruct`
            if let Some(composite_construct_operands) = components
                .into_iter()
                .map(|v| v.map(Operand::IdRef))
                .collect::<Option<Vec<_>>>()
            {
                // Leave all the other instructions in the chain as dead code for other passes
                // to clean up.
                *inst = Instruction::new(
                    Op::CompositeConstruct,
                    inst.result_type,
                    inst.result_id,
                    composite_construct_operands,
                );
            }
        }
    }
}

#[derive(Debug)]
enum IdentifiedOperand {
    /// The operand to the vectorized operation is a straight-up vector.
    Vector(Word),
    /// The operand to the vectorized operation is a collection of scalars that need to be packed
    /// together with `OpCompositeConstruct` before using the vectorized operation.
    Scalars(Vec<Word>),
    /// The operand to the vectorized operation is some non-value: for example, the `instruction`
    /// operand in `OpExtInst`.
    NonValue(Operand),
}

/// Given an ID ref to a `OpCompositeExtract`, get the vector it's extracting from, and the field
/// index.
fn get_composite_and_index(
    types: &FxHashMap<Word, Instruction>,
    defs: &FxHashMap<Word, Instruction>,
    id: Word,
    vector_width: u32,
) -> Option<(Word, u32)> {
    let inst = defs.get(&id)?;
    if inst.class.opcode != Op::CompositeExtract {
        return None;
    }
    if inst.operands.len() != 2 {
        // If the index is more than one deep, bail.
        return None;
    }
    let composite = inst.operands[0].unwrap_id_ref();
    let index = inst.operands[1].unwrap_literal_bit32();

    let composite_def = defs.get(&composite).or_else(|| types.get(&composite))?;
    let vector_def = types.get(&composite_def.result_type.unwrap())?;

    // Make sure it's a vector and has the width we're expecting.
    // Width mismatch would be doing something like `vec2(a.x + b.x, a.y + b.y)` where `a` is a
    // vec4 - if we optimized it to just `a + b`, it'd be incorrect.
    if vector_def.class.opcode != Op::TypeVector
        || vector_width != vector_def.operands[1].unwrap_literal_bit32()
    {
        return None;
    }

    Some((composite, index))
}

/// Given a bunch of operands (`results[n].operands[operand_index]`), where all those operands
/// refer to an `OpCompositeExtract` of the same vector (with proper indices, etc.), return that
/// vector.
fn match_vector_operand(
    types: &FxHashMap<Word, Instruction>,
    defs: &FxHashMap<Word, Instruction>,
    results: &[&Instruction],
    operand_index: usize,
    vector_width: u32,
) -> Option<Word> {
    let operand_zero = match results[0].operands[operand_index] {
        Operand::IdRef(id) => id,
        _ => {
            return None;
        }
    };
    // Extract the composite used for the first component.
    let composite_zero = match get_composite_and_index(types, defs, operand_zero, vector_width) {
        Some((composite_zero, 0)) => composite_zero,
        _ => {
            return None;
        }
    };
    // Check the same composite is used for every other component (and indices line up)
    for (expected_index, result) in results.iter().enumerate().skip(1) {
        let operand = match result.operands[operand_index] {
            Operand::IdRef(id) => id,
            _ => {
                return None;
            }
        };
        let (composite, actual_index) =
            match get_composite_and_index(types, defs, operand, vector_width) {
                Some(x) => x,
                None => {
                    return None;
                }
            };
        // If the source composite isn't all from the same composite, or the index
        // isn't the right index, break.
        if composite != composite_zero || expected_index != actual_index as usize {
            return None;
        }
    }
    Some(composite_zero)
}

/// Either extract out the vector behind each scalar component (see `match_vector_operand`), or
/// just return the collection of scalars for this operand (to be constructed into a vector via
/// `OpCompositeConstruct`).
fn match_vector_or_scalars_operand(
    types: &FxHashMap<Word, Instruction>,
    defs: &FxHashMap<Word, Instruction>,
    results: &[&Instruction],
    operand_index: usize,
    vector_width: u32,
) -> Option<IdentifiedOperand> {
    if let Some(composite) = match_vector_operand(types, defs, results, operand_index, vector_width)
    {
        Some(IdentifiedOperand::Vector(composite))
    } else {
        let operands = results
            .iter()
            .map(|inst| match inst.operands[operand_index] {
                Operand::IdRef(id) => Some(id),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()?;
        Some(IdentifiedOperand::Scalars(operands))
    }
}

/// Make sure all the operands are the same at this index, and return that operand. This is used
/// in, for example, the `instruction` operand for `OpExtInst`.
fn match_all_same_operand(results: &[&Instruction], operand_index: usize) -> Option<Operand> {
    let operand_zero = &results[0].operands[operand_index];
    if results
        .iter()
        .skip(1)
        .all(|inst| &inst.operands[operand_index] == operand_zero)
    {
        Some(operand_zero.clone())
    } else {
        None
    }
}

/// Find the proper operands for the vectorized operation. This means finding the backing vector
/// for each scalar component, etc.
fn match_operands(
    types: &FxHashMap<Word, Instruction>,
    defs: &FxHashMap<Word, Instruction>,
    results: &[&Instruction],
    vector_width: u32,
) -> Option<Vec<IdentifiedOperand>> {
    let operation_opcode = results[0].class.opcode;
    // Check to make sure they're all the same opcode, and have the same number of arguments.
    if results.iter().skip(1).any(|r| {
        r.class.opcode != operation_opcode || r.operands.len() != results[0].operands.len()
    }) {
        return None;
    }
    // TODO: There are probably other instructions relevant here.
    match operation_opcode {
        Op::IAdd
        | Op::FAdd
        | Op::ISub
        | Op::FSub
        | Op::IMul
        | Op::FMul
        | Op::UDiv
        | Op::SDiv
        | Op::FDiv
        | Op::UMod
        | Op::SRem
        | Op::FRem
        | Op::FMod
        | Op::ShiftRightLogical
        | Op::ShiftRightArithmetic
        | Op::ShiftLeftLogical
        | Op::BitwiseOr
        | Op::BitwiseXor
        | Op::BitwiseAnd => {
            let left = match_vector_or_scalars_operand(types, defs, results, 0, vector_width)?;
            let right = match_vector_or_scalars_operand(types, defs, results, 1, vector_width)?;
            match (left, right) {
                // Style choice: If all arguments are scalars, don't fuse this operation.
                (IdentifiedOperand::Scalars(_), IdentifiedOperand::Scalars(_)) => None,
                (left, right) => Some(vec![left, right]),
            }
        }
        Op::SNegate | Op::FNegate | Op::Not | Op::BitReverse => {
            let value = match_vector_operand(types, defs, results, 0, vector_width)?;
            Some(vec![IdentifiedOperand::Vector(value)])
        }
        Op::ExtInst => {
            let set = match_all_same_operand(results, 0)?;
            let instruction = match_all_same_operand(results, 1)?;
            let parameters = (2..results[0].operands.len())
                .map(|i| match_vector_or_scalars_operand(types, defs, results, i, vector_width));
            // Do some trickery to reduce allocations.
            let operands = IntoIterator::into_iter([
                Some(IdentifiedOperand::NonValue(set)),
                Some(IdentifiedOperand::NonValue(instruction)),
            ])
            .chain(parameters)
            .collect::<Option<Vec<_>>>()?;
            if operands
                .iter()
                .skip(2)
                .all(|p| matches!(p, &IdentifiedOperand::Scalars(_)))
            {
                // Style choice: If all arguments are scalars, don't fuse this operation.
                return None;
            }
            Some(operands)
        }
        _ => None,
    }
}

fn process_instruction(
    header: &mut ModuleHeader,
    types: &FxHashMap<Word, Instruction>,
    defs: &FxHashMap<Word, Instruction>,
    instructions: &mut Vec<Instruction>,
    instruction_index: &mut usize,
) -> Option<Instruction> {
    let inst = &instructions[*instruction_index];
    // Basic sanity checks
    if inst.class.opcode != Op::CompositeConstruct {
        return None;
    }
    let inst_result_id = inst.result_id.unwrap();
    let vector_ty = inst.result_type.unwrap();
    let vector_ty_inst = match types.get(&vector_ty) {
        Some(inst) => inst,
        _ => return None,
    };
    if vector_ty_inst.class.opcode != Op::TypeVector {
        return None;
    }
    let vector_width = vector_ty_inst.operands[1].unwrap_literal_bit32();
    // `results` is the defining instruction for each scalar component of the final result.
    let results = inst
        .operands
        .iter()
        .map(|op| defs.get(&op.unwrap_id_ref()))
        .collect::<Option<Vec<_>>>()?;

    let operation_opcode = results[0].class.opcode;
    // Figure out the operands for the vectorized instruction.
    let composite_arguments = match_operands(types, defs, &results, vector_width)?;

    // Fun little optimization: SPIR-V has a fancy OpVectorTimesScalar instruction. If we have a
    // vector times a collection of scalars, and the scalars are all the same, reduce it!
    if operation_opcode == Op::FMul
        && composite_arguments.len() == 2
        && let (&IdentifiedOperand::Vector(composite), IdentifiedOperand::Scalars(scalars))
        | (IdentifiedOperand::Scalars(scalars), &IdentifiedOperand::Vector(composite)) =
            (&composite_arguments[0], &composite_arguments[1])
    {
        let scalar = scalars[0];
        if scalars.iter().skip(1).all(|&s| s == scalar) {
            return Some(Instruction::new(
                Op::VectorTimesScalar,
                inst.result_type,
                inst.result_id,
                vec![Operand::IdRef(composite), Operand::IdRef(scalar)],
            ));
        }
    }

    // Map the operands into their concrete representations: vectors and non-values stay as-is, but
    // we need to emit an OpCompositeConstruct instruction for scalar collections.
    let operands = composite_arguments
        .into_iter()
        .map(|operand| match operand {
            IdentifiedOperand::Vector(composite) => Operand::IdRef(composite),
            IdentifiedOperand::NonValue(operand) => operand,
            IdentifiedOperand::Scalars(scalars) => {
                let id = super::id(header);
                // spirv-opt will transform this into an OpConstantComposite if all arguments are
                // constant, so we don't have to worry about that.
                instructions.insert(
                    *instruction_index,
                    Instruction::new(
                        Op::CompositeConstruct,
                        Some(vector_ty),
                        Some(id),
                        scalars.into_iter().map(Operand::IdRef).collect(),
                    ),
                );
                *instruction_index += 1;
                Operand::IdRef(id)
            }
        })
        .collect();

    Some(Instruction::new(
        operation_opcode,
        Some(vector_ty),
        Some(inst_result_id),
        operands,
    ))
}

/// Fuse a sequence of scalar operations into a single vector operation. For example:
/// ```
/// %x_0 = OpCompositeExtract %x 0
/// %x_1 = OpCompositeExtract %x 1
/// %y_0 = OpCompositeExtract %y 0
/// %y_1 = OpCompositeExtract %y 1
/// %r_0 = OpAdd %x_0 %y_0
/// %r_1 = OpAdd %x_1 %y_1
/// %r = OpCompositeConstruct %r_0 %r_1
/// ```
/// into
/// ```
/// %r = OpAdd %x %y
/// ```
/// (We don't remove the intermediate instructions, however, in case they're used elsewhere - we
/// let spirv-opt remove them if they're actually dead)
pub fn vector_ops(
    header: &mut ModuleHeader,
    types: &FxHashMap<Word, Instruction>,
    function: &mut Function,
) {
    let defs = function
        .all_inst_iter()
        .filter_map(|inst| Some((inst.result_id?, inst.clone())))
        .collect::<FxHashMap<Word, Instruction>>();
    for block in &mut function.blocks {
        // It'd be nice to iterate over &mut block.instructions, but there's a weird case: if we
        // have a vector plus a collection of scalars, we want to pack the collection of scalars
        // into a vector and do a vector+vector op. That means we need to insert an extra
        // OpCompositeConstruct into the block, so, we need to manually keep track of the current
        // index and do a while loop.
        let mut instruction_index = 0;
        while instruction_index < block.instructions.len() {
            if let Some(result) = process_instruction(
                header,
                types,
                &defs,
                &mut block.instructions,
                &mut instruction_index,
            ) {
                // Leave all the other instructions in the chain as dead code for other passes
                // to clean up.
                block.instructions[instruction_index] = result;
            }

            instruction_index += 1;
        }
    }
}

fn can_fuse_bool(
    types: &FxHashMap<Word, Instruction>,
    defs: &FxHashMap<Word, (usize, Instruction)>,
    inst: &Instruction,
) -> bool {
    fn constant_value(types: &FxHashMap<Word, Instruction>, val: Word) -> Option<u32> {
        let inst = types.get(&val)?;
        if inst.class.opcode != Op::Constant {
            return None;
        }
        match inst.operands[0] {
            Operand::LiteralBit32(v) => Some(v),
            _ => None,
        }
    }

    fn visit(
        types: &FxHashMap<Word, Instruction>,
        defs: &FxHashMap<Word, (usize, Instruction)>,
        visited: &mut FxHashSet<Word>,
        value: Word,
    ) -> bool {
        if visited.insert(value) {
            let inst = match defs.get(&value) {
                Some((_, inst)) => inst,
                None => return false,
            };
            match inst.class.opcode {
                Op::Select => {
                    constant_value(types, inst.operands[1].unwrap_id_ref()) == Some(1)
                        && constant_value(types, inst.operands[2].unwrap_id_ref()) == Some(0)
                }
                Op::Phi => inst
                    .operands
                    .iter()
                    .step_by(2)
                    .all(|op| visit(types, defs, visited, op.unwrap_id_ref())),
                _ => false,
            }
        } else {
            true
        }
    }

    if inst.class.opcode != Op::INotEqual
        || constant_value(types, inst.operands[1].unwrap_id_ref()) != Some(0)
    {
        return false;
    }
    let int_value = inst.operands[0].unwrap_id_ref();

    visit(types, defs, &mut FxHashSet::default(), int_value)
}

fn fuse_bool(
    header: &mut ModuleHeader,
    defs: &FxHashMap<Word, (usize, Instruction)>,
    phis_to_insert: &mut Vec<(usize, Instruction)>,
    already_mapped: &mut FxHashMap<Word, Word>,
    bool_ty: Word,
    int_value: Word,
) -> Word {
    if let Some(&result) = already_mapped.get(&int_value) {
        return result;
    }
    let (block_of_inst, inst) = defs.get(&int_value).unwrap();
    match inst.class.opcode {
        Op::Select => inst.operands[0].unwrap_id_ref(),
        Op::Phi => {
            let result_id = id(header);
            already_mapped.insert(int_value, result_id);
            let new_phi_args = inst
                .operands
                .chunks(2)
                .flat_map(|arr| {
                    let phi_value = &arr[0];
                    let block = &arr[1];
                    [
                        Operand::IdRef(fuse_bool(
                            header,
                            defs,
                            phis_to_insert,
                            already_mapped,
                            bool_ty,
                            phi_value.unwrap_id_ref(),
                        )),
                        block.clone(),
                    ]
                })
                .collect::<Vec<_>>();
            let inst = Instruction::new(Op::Phi, Some(bool_ty), Some(result_id), new_phi_args);
            phis_to_insert.push((*block_of_inst, inst));
            result_id
        }
        _ => bug!("can_fuse_bool should have prevented this case"),
    }
}

// The compiler generates a lot of code that looks like this:
// %v_int = OpSelect %int %v %const_1 %const_0
// %v2 = OpINotEqual %bool %v_int %const_0
// (This is due to rustc/spirv not supporting bools in memory, and needing to convert to u8, but
// then things get inlined/mem2reg'd)
//
// This pass fuses together those two instructions to strip out the intermediate integer variable.
// The purpose is to make simple code that doesn't actually do memory-stuff with bools not require
// the Int8 capability (and so we can't rely on spirv-opt to do this same pass).
//
// Unfortunately, things get complicated because of phis: the majority of actually useful cases to
// do this pass need to track pseudo-bool ints through phi instructions.
//
// The logic goes like:
// 1) Figure out what we *can* fuse. This means finding OpINotEqual instructions (converting back
//    from int->bool) and tracing the value back recursively through any phis, and making sure each
//    one terminates in either a loop back around to something we've already seen, or an OpSelect
//    (converting from bool->int).
// 2) Do the fusion. Trace back through phis, generating a second bool-typed phi alongside the
//    original int-typed phi, and when hitting an OpSelect, taking the bool value directly.
// 3) DCE the dead OpSelects/int-typed OpPhis (done in a later pass). We don't nuke them here,
//    since they might be used elsewhere, and don't want to accidentally leave a dangling
//    reference.
pub fn bool_fusion(
    header: &mut ModuleHeader,
    types: &FxHashMap<Word, Instruction>,
    function: &mut Function,
) {
    let defs: FxHashMap<Word, (usize, Instruction)> = function
        .blocks
        .iter()
        .enumerate()
        .flat_map(|(block_id, block)| {
            block
                .instructions
                .iter()
                .filter_map(move |inst| Some((inst.result_id?, (block_id, inst.clone()))))
        })
        .collect();
    let mut rewrite_rules = FxHashMap::default();
    let mut phis_to_insert = Default::default();
    let mut already_mapped = Default::default();
    for block in &mut function.blocks {
        for inst in &mut block.instructions {
            if can_fuse_bool(types, &defs, inst) {
                let rewrite_to = fuse_bool(
                    header,
                    &defs,
                    &mut phis_to_insert,
                    &mut already_mapped,
                    inst.result_type.unwrap(),
                    inst.operands[0].unwrap_id_ref(),
                );
                rewrite_rules.insert(inst.result_id.unwrap(), rewrite_to);
                *inst = Instruction::new(Op::Nop, None, None, Vec::new());
            }
        }
    }
    for (block, phi) in phis_to_insert {
        function.blocks[block].instructions.insert(0, phi);
    }
    super::apply_rewrite_rules(&rewrite_rules, &mut function.blocks);
}

/// Fold `OpLoad` from Private/Function storage class variables that have constant initializers.
///
/// This optimization handles patterns like:
/// ```text
/// %ptr = OpVariable %_ptr_Private_T Private %initializer
/// %val = OpLoad %T %ptr
/// ```
/// After this optimization, uses of `%val` are replaced with `%initializer`.
///
/// This is particularly important for pointer-to-pointer constants (e.g., `&&123`)
/// where the outer pointer variable has a known initializer (the inner pointer).
/// Without this optimization, such patterns generate invalid SPIR-V in Logical
/// addressing mode (pointer-to-pointer with Private storage class is not allowed).
pub fn fold_load_from_constant_variable(module: &mut Module) {
    use rspirv::spirv::StorageClass;

    // Build a map of variable ID -> initializer ID for Private/Function variables
    // that have constant initializers.
    let var_initializers: FxHashMap<Word, Word> = module
        .types_global_values
        .iter()
        .filter_map(|inst| {
            if inst.class.opcode != Op::Variable {
                return None;
            }
            // Check storage class - only Private and Function can be folded
            let storage_class = inst.operands.first()?.unwrap_storage_class();
            if !matches!(
                storage_class,
                StorageClass::Private | StorageClass::Function
            ) {
                return None;
            }
            // Check for initializer (second operand after storage class)
            let initializer = inst.operands.get(1)?.id_ref_any()?;
            Some((inst.result_id?, initializer))
        })
        .collect();

    if var_initializers.is_empty() {
        return;
    }

    // Rewrite OpLoad instructions that load from variables with known initializers
    let mut rewrite_rules: FxHashMap<Word, Word> = FxHashMap::default();

    for func in &mut module.functions {
        for block in &mut func.blocks {
            for inst in &mut block.instructions {
                if inst.class.opcode != Op::Load {
                    continue;
                }
                // OpLoad has pointer as first operand
                let ptr_id = inst.operands[0].unwrap_id_ref();
                if let Some(&initializer) = var_initializers.get(&ptr_id) {
                    // This load can be replaced with the initializer value
                    if let Some(result_id) = inst.result_id {
                        rewrite_rules.insert(result_id, initializer);
                        // Turn this instruction into a Nop
                        *inst = Instruction::new(Op::Nop, None, None, Vec::new());
                    }
                }
            }
        }
    }

    if !rewrite_rules.is_empty() {
        // Apply rewrite rules to all functions
        for func in &mut module.functions {
            super::apply_rewrite_rules(&rewrite_rules, &mut func.blocks);
        }
    }
}

/// Replace each pointer-typed `OpPhi` whose incoming values are all one value (or the phi
/// itself, around a loop) with that value. mem2reg leaves these when a pointer only passes
/// through block parameters; logical addressing forbids pointer phis anyway.
fn fold_trivial_pointer_phis(
    func: &mut Function,
    pointee: &FxHashMap<Word, (rspirv::spirv::StorageClass, Word)>,
) {
    loop {
        let mut replace = FxHashMap::default();
        for inst in func.all_inst_iter() {
            if inst.class.opcode != Op::Phi
                || !inst.result_type.is_some_and(|ty| pointee.contains_key(&ty))
            {
                continue;
            }
            let phi = inst.result_id.unwrap();
            let mut incoming = inst
                .operands
                .chunks(2)
                .map(|pair| pair[0].unwrap_id_ref())
                .filter(|&v| v != phi);
            let Some(first) = incoming.next() else {
                continue;
            };
            if incoming.all(|v| v == first) {
                replace.insert(phi, first);
            }
        }
        if replace.is_empty() {
            return;
        }
        let resolve = |mut id: Word| {
            while let Some(&next) = replace.get(&id) {
                id = next;
            }
            id
        };
        for block in &mut func.blocks {
            block
                .instructions
                .retain(|inst| !inst.result_id.is_some_and(|id| replace.contains_key(&id)));
            for inst in &mut block.instructions {
                for op in &mut inst.operands {
                    if let Operand::IdRef(id) = op {
                        *id = resolve(*id);
                    }
                }
            }
        }
    }
}

/// Rewrite access chains through pointer `OpBitcast`s left by `pointercast` when it could not
/// express a cast as an access chain (`SpirvValueKind::LogicalPtrCast`), so that slices made
/// from arrays and sub-slices of slices become legal once inlining has put the cast and its
/// uses in one function:
/// - `*[T; N] as *[T]` (array unsizing): `AccessChain(cast, i, ..)` indexes the array directly,
///   `AccessChain(p, i, ..)`.
/// - `*T as *[T]` where `*T` is itself `AccessChain(base, .., j)` into an array of `T`
///   (`slice_from_raw_parts(&s[j], ..)`): `AccessChain(cast, i, ..)` becomes
///   `AccessChain(base, .., j + i, ..)`.
///
/// Casts left unused are removed by the DCE that follows; any that survive keep their zombie
/// and are reported by the late diagnostics.
pub fn legalize_ptr_bitcasts(module: &mut Module) {
    let Module {
        header,
        types_global_values,
        functions,
        ..
    } = module;
    let header = header.as_mut().unwrap();
    let mut types = PtrTypes::new(types_global_values);
    for func in functions.iter_mut() {
        loop {
            fold_trivial_pointer_phis(func, &types.pointee);
            let split = split_pointer_aggregate_phis(func, &types, header);
            if split {
                super::destructure_composites::destructure_composites(func);
                fold_trivial_pointer_phis(func, &types.pointee);
            }
            let indexed = index_pointer_phis(func, &types, header);
            let mut defs = FxHashMap::default();
            let mut func_result_type = types.result_type.clone();
            for inst in func.parameters.iter().chain(func.all_inst_iter()) {
                if let Some(id) = inst.result_id {
                    defs.insert(id, inst.clone());
                    if let Some(ty) = inst.result_type {
                        func_result_type.insert(id, ty);
                    }
                }
            }

            let mut changed = false;
            let mut retype = FxHashMap::default();
            for block in &mut func.blocks {
                let mut i = 0;
                while i < block.instructions.len() {
                    let inst = &block.instructions[i];
                    i += 1;
                    if !matches!(inst.class.opcode, Op::AccessChain | Op::InBoundsAccessChain)
                        || inst.operands.len() < 2
                    {
                        continue;
                    }
                    let Some(cast) = defs
                        .get(&inst.operands[0].unwrap_id_ref())
                        .filter(|d| d.class.opcode == Op::Bitcast)
                    else {
                        continue;
                    };
                    let src = cast.operands[0].unwrap_id_ref();
                    let (Some(&(src_class, src_pointee)), Some(&(_, dst_pointee))) = (
                        func_result_type
                            .get(&src)
                            .and_then(|ty| types.pointee.get(ty)),
                        cast.result_type.and_then(|ty| types.pointee.get(&ty)),
                    ) else {
                        continue;
                    };
                    let Some(&elem) = types.array_elem.get(&dst_pointee) else {
                        continue;
                    };
                    let result = inst.result_id.unwrap();

                    if types.array_elem.get(&src_pointee) == Some(&elem) {
                        block.instructions[i - 1].operands[0] = Operand::IdRef(src);
                        retype.insert(result, src_class);
                        changed = true;
                        continue;
                    }

                    let Some(outer) = defs.get(&src).filter(|d| {
                        src_pointee == elem
                            && matches!(d.class.opcode, Op::AccessChain | Op::InBoundsAccessChain)
                            && d.operands.len() >= 2
                    }) else {
                        continue;
                    };
                    let outer_last = outer.operands.last().unwrap().unwrap_id_ref();
                    let first = inst.operands[1].unwrap_id_ref();
                    let int_ty = func_result_type.get(&outer_last).copied();
                    if int_ty.is_none() || int_ty != func_result_type.get(&first).copied() {
                        continue;
                    }
                    let sum = super::id(header);
                    let add = Instruction::new(
                        Op::IAdd,
                        int_ty,
                        Some(sum),
                        vec![Operand::IdRef(outer_last), Operand::IdRef(first)],
                    );
                    let opcode = if outer.class.opcode == Op::InBoundsAccessChain
                        && inst.class.opcode == Op::InBoundsAccessChain
                    {
                        Op::InBoundsAccessChain
                    } else {
                        Op::AccessChain
                    };
                    let mut operands = outer.operands[..outer.operands.len() - 1].to_vec();
                    operands.push(Operand::IdRef(sum));
                    operands.extend_from_slice(&inst.operands[2..]);
                    block.instructions[i - 1] =
                        Instruction::new(opcode, inst.result_type, Some(result), operands);
                    block.instructions.insert(i - 1, add);
                    retype.insert(result, src_class);
                    changed = true;
                    i += 1;
                }
            }
            if !changed && !split && !indexed {
                break;
            }
            retype_derived_pointers(func, retype, &mut types, header);
            super::destructure_composites::destructure_composites(func);
        }
    }
    types_global_values.extend(types.created);
}

/// Pointer types by storage class and pointee, creating the ones a rewrite needs.
struct PtrTypes {
    pointee: FxHashMap<Word, (rspirv::spirv::StorageClass, Word)>,
    members: FxHashMap<Word, Vec<Word>>,
    arrays: FxHashMap<Word, (Word, u32)>,
    zero_u32: Option<Word>,
    by_class_and_pointee: FxHashMap<(rspirv::spirv::StorageClass, Word), Word>,
    array_elem: FxHashMap<Word, Word>,
    result_type: FxHashMap<Word, Word>,
    created: Vec<Instruction>,
}

impl PtrTypes {
    fn new(types_global_values: &[Instruction]) -> Self {
        let mut types = Self {
            pointee: FxHashMap::default(),
            members: FxHashMap::default(),
            arrays: FxHashMap::default(),
            zero_u32: None,
            by_class_and_pointee: FxHashMap::default(),
            array_elem: FxHashMap::default(),
            result_type: FxHashMap::default(),
            created: Vec::new(),
        };
        for inst in types_global_values {
            let Some(id) = inst.result_id else { continue };
            match inst.class.opcode {
                Op::TypePointer => {
                    let key = (
                        inst.operands[0].unwrap_storage_class(),
                        inst.operands[1].unwrap_id_ref(),
                    );
                    types.pointee.insert(id, key);
                    types.by_class_and_pointee.entry(key).or_insert(id);
                }
                Op::TypeArray | Op::TypeRuntimeArray => {
                    types
                        .array_elem
                        .insert(id, inst.operands[0].unwrap_id_ref());
                }
                Op::TypeStruct => {
                    types.members.insert(
                        id,
                        inst.operands.iter().map(|op| op.unwrap_id_ref()).collect(),
                    );
                }
                _ => {}
            }
            if let Some(ty) = inst.result_type {
                types.result_type.insert(id, ty);
            }
        }
        let constants: FxHashMap<Word, u32> = types_global_values
            .iter()
            .filter(|inst| inst.class.opcode == Op::Constant)
            .filter_map(|inst| match inst.operands[0] {
                Operand::LiteralBit32(v) => Some((inst.result_id?, v)),
                _ => None,
            })
            .collect();
        let u32_ty = types_global_values
            .iter()
            .find(|inst| {
                inst.class.opcode == Op::TypeInt
                    && inst.operands == [Operand::LiteralBit32(32), Operand::LiteralBit32(0)]
            })
            .and_then(|inst| inst.result_id);
        types.zero_u32 = types_global_values
            .iter()
            .find(|inst| {
                inst.class.opcode == Op::Constant
                    && inst.result_type == u32_ty
                    && u32_ty.is_some()
                    && inst.operands == [Operand::LiteralBit32(0)]
            })
            .and_then(|inst| inst.result_id);
        for inst in types_global_values {
            if inst.class.opcode == Op::TypeArray
                && let Some(&len) = constants.get(&inst.operands[1].unwrap_id_ref())
            {
                types.arrays.insert(
                    inst.result_id.unwrap(),
                    (inst.operands[0].unwrap_id_ref(), len),
                );
            }
        }
        types
    }

    /// Whether `ty` holds no data (structs of nothing but such structs, or empty arrays).
    fn is_empty(&self, ty: Word) -> bool {
        self.members
            .get(&ty)
            .is_some_and(|m| m.iter().all(|&t| self.is_empty(t)))
            || self
                .arrays
                .get(&ty)
                .is_some_and(|&(e, len)| len == 0 || self.is_empty(e))
    }

    fn contains_pointer(&self, ty: Word) -> bool {
        self.pointee.contains_key(&ty)
            || self
                .members
                .get(&ty)
                .is_some_and(|m| m.iter().any(|&t| self.contains_pointer(t)))
            || self
                .arrays
                .get(&ty)
                .is_some_and(|&(e, _)| self.contains_pointer(e))
    }

    /// The parts of `ty` a pointer-free value is made of: pointers, and the largest pointer-free
    /// aggregates, with their access paths. Aggregates holding a pointer are split.
    fn pointer_leaves(&self, ty: Word, path: &mut Vec<u32>, out: &mut Vec<(Vec<u32>, Word)>) {
        if self.pointee.contains_key(&ty) || !self.contains_pointer(ty) {
            out.push((path.clone(), ty));
        } else if let Some(members) = self.members.get(&ty) {
            for (i, &member) in members.iter().enumerate() {
                path.push(i as u32);
                self.pointer_leaves(member, path, out);
                path.pop();
            }
        } else if let Some(&(elem, len)) = self.arrays.get(&ty) {
            for i in 0..len {
                path.push(i);
                self.pointer_leaves(elem, path, out);
                path.pop();
            }
        } else {
            out.push((path.clone(), ty));
        }
    }

    fn pointer(
        &mut self,
        class: rspirv::spirv::StorageClass,
        pointee: Word,
        header: &mut ModuleHeader,
    ) -> Word {
        if let Some(&id) = self.by_class_and_pointee.get(&(class, pointee)) {
            return id;
        }
        let id = super::id(header);
        self.created.push(Instruction::new(
            Op::TypePointer,
            None,
            Some(id),
            vec![Operand::StorageClass(class), Operand::IdRef(pointee)],
        ));
        self.pointee.insert(id, (class, pointee));
        self.by_class_and_pointee.insert((class, pointee), id);
        id
    }
}

/// Give each pointer in `retype`, and every pointer derived from one through access chains,
/// copies or phis, a type in its storage class. A cast left these typed in the class
/// inference guessed for the cast, which no longer matches the memory they point into.
fn retype_derived_pointers(
    func: &mut Function,
    mut retype: FxHashMap<Word, rspirv::spirv::StorageClass>,
    types: &mut PtrTypes,
    header: &mut ModuleHeader,
) {
    loop {
        let mut grew = false;
        for inst in func.all_inst_iter() {
            let Some(result) = inst.result_id else {
                continue;
            };
            if retype.contains_key(&result)
                || !inst
                    .result_type
                    .is_some_and(|ty| types.pointee.contains_key(&ty))
            {
                continue;
            }
            let from = match inst.class.opcode {
                Op::AccessChain | Op::InBoundsAccessChain | Op::CopyObject => {
                    retype.get(&inst.operands[0].unwrap_id_ref()).copied()
                }
                Op::Phi => inst
                    .operands
                    .chunks(2)
                    .find_map(|pair| retype.get(&pair[0].unwrap_id_ref()).copied()),
                _ => None,
            };
            if let Some(class) = from {
                retype.insert(result, class);
                grew = true;
            }
        }
        if !grew {
            break;
        }
    }
    for block in &mut func.blocks {
        for inst in &mut block.instructions {
            let (Some(result), Some(ty)) = (inst.result_id, inst.result_type) else {
                continue;
            };
            let Some(&class) = retype.get(&result) else {
                continue;
            };
            let Some(&(old_class, pointee)) = types.pointee.get(&ty) else {
                continue;
            };
            if old_class != class {
                inst.result_type = Some(types.pointer(class, pointee, header));
            }
        }
    }
}

/// Turn each conditional branch with exactly one side that can only reach `OpUnreachable` into
/// an unconditional branch to the other side. Reaching `OpUnreachable` is undefined behavior,
/// so the condition may be assumed to hold; panics outside entry points lower to it (unless an
/// abort strategy says otherwise), so this removes bounds and overflow checks from the code
/// drivers schedule. A check left between memory operations keeps drivers from issuing them
/// together (measured on NVIDIA: a register-blocked cooperative-matrix GEMM ran ~4.5× slower).
///
/// A block can only reach `OpUnreachable` when every instruction in it is free of side effects
/// and every successor can only reach `OpUnreachable` too, as on the paths core's panic helpers
/// take to the abort.
pub fn fold_branches_to_unreachable(module: &mut Module) {
    let pure_ext_sets: FxHashSet<Word> = module
        .ext_inst_imports
        .iter()
        .filter(|inst| {
            let name = inst.operands[0].unwrap_literal_string();
            name == "GLSL.std.450" || name.starts_with("NonSemantic.Shader.DebugInfo")
        })
        .map(|inst| inst.result_id.unwrap())
        .collect();
    let is_pure = |inst: &Instruction| match inst.class.opcode {
        Op::Line | Op::NoLine | Op::SelectionMerge | Op::Phi => true,
        Op::ExtInst => pure_ext_sets.contains(&inst.operands[0].unwrap_id_ref()),
        Op::FunctionCall | Op::LoopMerge => false,
        op => inst.result_id.is_some() && !format!("{op:?}").starts_with("Atomic"),
    };

    for func in &mut module.functions {
        let mut dead = FxHashSet::default();
        loop {
            let before = dead.len();
            for block in &func.blocks {
                let Some((terminator, body)) = block.instructions.split_last() else {
                    continue;
                };
                let label = block.label_id().unwrap();
                if dead.contains(&label) || !body.iter().all(is_pure) {
                    continue;
                }
                let targets_dead = |ops: &[Operand]| {
                    ops.iter().all(|op| match op {
                        Operand::IdRef(id) => dead.contains(id),
                        _ => true,
                    })
                };
                let is_dead = match terminator.class.opcode {
                    Op::Unreachable => true,
                    Op::Branch => dead.contains(&terminator.operands[0].unwrap_id_ref()),
                    Op::BranchConditional => targets_dead(&terminator.operands[1..3]),
                    Op::Switch => targets_dead(&terminator.operands[1..]),
                    _ => false,
                };
                if is_dead {
                    dead.insert(label);
                }
            }
            if dead.len() == before {
                break;
            }
        }
        if dead.is_empty() {
            continue;
        }

        for block in &mut func.blocks {
            if dead.contains(&block.label_id().unwrap()) {
                continue;
            }
            let Some(terminator) = block.instructions.last() else {
                continue;
            };
            if terminator.class.opcode != Op::BranchConditional {
                continue;
            }
            let (if_true, if_false) = (
                terminator.operands[1].unwrap_id_ref(),
                terminator.operands[2].unwrap_id_ref(),
            );
            let live = match (dead.contains(&if_true), dead.contains(&if_false)) {
                (false, true) => if_true,
                (true, false) => if_false,
                _ => continue,
            };
            let len = block.instructions.len();
            block.instructions[len - 1] =
                Instruction::new(Op::Branch, None, None, vec![Operand::IdRef(live)]);
            if len >= 2 && block.instructions[len - 2].class.opcode == Op::SelectionMerge {
                block.instructions.remove(len - 2);
            }
        }
    }
}

/// Insert `insts` at the end of `block`, before its terminator and any merge instruction.
fn insert_before_terminator(block: &mut rspirv::dr::Block, insts: Vec<Instruction>) {
    let mut at = block.instructions.len() - 1;
    if at > 0
        && matches!(
            block.instructions[at - 1].class.opcode,
            Op::SelectionMerge | Op::LoopMerge
        )
    {
        at -= 1;
    }
    block.instructions.splice(at..at, insts);
}

fn replace_uses(func: &mut Function, from: Word, to: Word) {
    for inst in func.all_inst_iter_mut() {
        for op in &mut inst.operands {
            if *op == Operand::IdRef(from) {
                *op = Operand::IdRef(to);
            }
        }
    }
}

/// Split each `OpPhi` of an aggregate type holding a pointer into one phi per part (see
/// [`PtrTypes::pointer_leaves`]), extracting the parts at the end of each predecessor and
/// rebuilding the aggregate after the block's phis. Logical addressing forbids pointers inside
/// aggregates; once split, the pointer phis can be legalized on their own.
fn split_pointer_aggregate_phis(
    func: &mut Function,
    types: &PtrTypes,
    header: &mut ModuleHeader,
) -> bool {
    let mut changed = false;
    loop {
        let found = func.blocks.iter().enumerate().find_map(|(b, block)| {
            block
                .instructions
                .iter()
                .take_while(|inst| inst.class.opcode == Op::Phi)
                .position(|inst| {
                    let ty = inst.result_type.unwrap();
                    !types.pointee.contains_key(&ty) && types.contains_pointer(ty)
                })
                .map(|i| (b, i))
        });
        let Some((b, i)) = found else { return changed };
        changed = true;
        let phi = func.blocks[b].instructions.remove(i);
        let phi_id = phi.result_id.unwrap();
        let ty = phi.result_type.unwrap();
        let mut leaves = Vec::new();
        types.pointer_leaves(ty, &mut Vec::new(), &mut leaves);

        let label_to_block: FxHashMap<Word, usize> = func
            .blocks
            .iter()
            .enumerate()
            .map(|(i, block)| (block.label_id().unwrap(), i))
            .collect();
        let incoming: Vec<(Word, Word)> = phi
            .operands
            .chunks(2)
            .map(|pair| (pair[0].unwrap_id_ref(), pair[1].unwrap_id_ref()))
            .collect();
        let leaf_ids: Vec<Word> = leaves.iter().map(|_| super::id(header)).collect();
        // Parts that hold no data need no phi; nothing ever writes them, so tracing them back
        // through the incoming values would only end at an `OpUndef` and keep the rest alive.
        let empty: Vec<bool> = leaves
            .iter()
            .map(|&(_, leaf_ty)| types.is_empty(leaf_ty))
            .collect();
        let mut leaf_operands: Vec<Vec<Operand>> = vec![Vec::new(); leaves.len()];
        for &(value, pred) in &incoming {
            let mut extracts = Vec::new();
            for (l, (path, leaf_ty)) in leaves.iter().enumerate() {
                if empty[l] {
                    continue;
                }
                let part = if value == phi_id {
                    leaf_ids[l]
                } else {
                    let id = super::id(header);
                    let mut operands = vec![Operand::IdRef(value)];
                    operands.extend(path.iter().map(|&i| Operand::LiteralBit32(i)));
                    extracts.push(Instruction::new(
                        Op::CompositeExtract,
                        Some(*leaf_ty),
                        Some(id),
                        operands,
                    ));
                    id
                };
                leaf_operands[l].extend([Operand::IdRef(part), Operand::IdRef(pred)]);
            }
            if let Some(&pb) = label_to_block.get(&pred) {
                insert_before_terminator(&mut func.blocks[pb], extracts);
            }
        }

        let mut rebuild: Vec<Instruction> = leaves
            .iter()
            .zip(&leaf_ids)
            .zip(&empty)
            .filter(|(_, is_empty)| **is_empty)
            .map(|(((_, leaf_ty), &id), _)| {
                Instruction::new(Op::Undef, Some(*leaf_ty), Some(id), vec![])
            })
            .collect();
        let root = build_from_leaves(
            types,
            ty,
            &mut Vec::new(),
            &leaves,
            &leaf_ids,
            &mut rebuild,
            header,
        );
        let block = &mut func.blocks[b];
        let new_phis = leaves
            .iter()
            .zip(&leaf_ids)
            .zip(leaf_operands)
            .zip(&empty)
            .filter(|(_, is_empty)| !**is_empty)
            .map(|((((_, leaf_ty), &id), operands), _)| {
                Instruction::new(Op::Phi, Some(*leaf_ty), Some(id), operands)
            });
        block.instructions.splice(i..i, new_phis);
        let after_phis = block
            .instructions
            .iter()
            .take_while(|inst| inst.class.opcode == Op::Phi)
            .count();
        block.instructions.splice(after_phis..after_phis, rebuild);
        replace_uses(func, phi_id, root);
    }
}

/// The value of type `ty` at `path` rebuilt from `leaves`, pushing the constructs onto `out`.
fn build_from_leaves(
    types: &PtrTypes,
    ty: Word,
    path: &mut Vec<u32>,
    leaves: &[(Vec<u32>, Word)],
    leaf_ids: &[Word],
    out: &mut Vec<Instruction>,
    header: &mut ModuleHeader,
) -> Word {
    if let Some(l) = leaves.iter().position(|(p, _)| p == path) {
        return leaf_ids[l];
    }
    let parts: Vec<Word> = if let Some(members) = types.members.get(&ty) {
        members.clone()
    } else {
        let (elem, len) = types.arrays[&ty];
        vec![elem; len as usize]
    };
    let mut constituents = Vec::new();
    for (i, part_ty) in parts.into_iter().enumerate() {
        path.push(i as u32);
        constituents.push(Operand::IdRef(build_from_leaves(
            types, part_ty, path, leaves, leaf_ids, out, header,
        )));
        path.pop();
    }
    let id = super::id(header);
    out.push(Instruction::new(
        Op::CompositeConstruct,
        Some(ty),
        Some(id),
        constituents,
    ));
    id
}

/// One incoming pointer of a phi `index_pointer_phis` rewrites, as an element pointer
/// `AccessChain(prefix.., index)`. `index` is `Offset(off)` for a re-slice of the phi itself
/// (`&prev[off..]`): the phi's own index plus `off`.
struct Incoming {
    chain_ty: Word,
    prefix: Vec<Operand>,
    inbounds: bool,
    index: IncomingIndex,
}

#[derive(Clone, Copy)]
enum IncomingIndex {
    At(Word),
    Offset(Word),
    Same,
}

/// Rewrite each pointer `OpPhi` whose incoming pointers all address one array, differing only in
/// the element they start at, into one `AccessChain` over a phi of that index. Logical addressing
/// forbids choosing between pointers at run time; a loop that carries a slice (or a struct or
/// array holding one, once split) from step to step needs exactly this.
///
/// For a slice phi (a pointer to an array), each incoming slice may be a whole array or slice
/// (index 0), a sub-slice `Bitcast(AccessChain(.., i))` (index `i`), or a sub-slice of the phi
/// itself (its index plus the offset); the result is the bitcast of the new access chain. For an
/// element pointer phi, each incoming value is an `AccessChain(.., i)`.
fn index_pointer_phis(func: &mut Function, types: &PtrTypes, header: &mut ModuleHeader) -> bool {
    let mut changed = false;
    loop {
        let defs: FxHashMap<Word, Instruction> = func
            .all_inst_iter()
            .filter_map(|inst| Some((inst.result_id?, inst.clone())))
            .collect();
        let type_of = |id: Word| {
            defs.get(&id)
                .and_then(|d| d.result_type)
                .or_else(|| types.result_type.get(&id).copied())
        };

        let mut plan = None;
        'search: for (b, block) in func.blocks.iter().enumerate() {
            for (i, phi) in block
                .instructions
                .iter()
                .enumerate()
                .take_while(|(_, inst)| inst.class.opcode == Op::Phi)
            {
                let Some(&(_, phi_pointee)) = phi.result_type.and_then(|ty| types.pointee.get(&ty))
                else {
                    continue;
                };
                let phi_id = phi.result_id.unwrap();
                let slice_elem = types.array_elem.get(&phi_pointee).copied();
                let incoming: Option<Vec<(Option<Incoming>, Word)>> = phi
                    .operands
                    .chunks(2)
                    .map(|pair| {
                        let (value, pred) = (pair[0].unwrap_id_ref(), pair[1].unwrap_id_ref());
                        if value == phi_id {
                            return Some((None, pred));
                        }
                        let desc = describe_incoming(
                            value,
                            phi_id,
                            slice_elem,
                            phi.result_type.unwrap(),
                            &defs,
                            types,
                            &type_of,
                        )?;
                        Some((Some(desc), pred))
                    })
                    .collect();
                let Some(incoming) = incoming else { continue };
                let described: Vec<&Incoming> =
                    incoming.iter().filter_map(|(d, _)| d.as_ref()).collect();
                let Some(anchor) = described
                    .iter()
                    .find(|d| matches!(d.index, IncomingIndex::At(_)))
                else {
                    continue;
                };
                let IncomingIndex::At(anchor_index) = anchor.index else {
                    unreachable!()
                };
                let idx_ty = type_of(anchor_index);
                let index_ty_matches = |d: &Incoming| match d.index {
                    IncomingIndex::At(x) | IncomingIndex::Offset(x) => type_of(x) == idx_ty,
                    IncomingIndex::Same => true,
                };
                let consistent = described.iter().all(|d| {
                    d.chain_ty == anchor.chain_ty
                        && (matches!(d.index, IncomingIndex::Offset(_))
                            || d.prefix == anchor.prefix)
                        && index_ty_matches(d)
                });
                if !consistent || idx_ty.is_none() {
                    continue;
                }
                let inbounds = described.iter().all(|d| d.inbounds);
                let indices: Vec<(IncomingIndex, Word)> = incoming
                    .iter()
                    .map(|(d, pred)| (d.as_ref().map_or(IncomingIndex::Same, |d| d.index), *pred))
                    .collect();
                let cast = slice_elem.map(|_| phi.result_type.unwrap());
                plan = Some((
                    b,
                    i,
                    phi_id,
                    cast,
                    anchor.chain_ty,
                    anchor.prefix.clone(),
                    inbounds,
                    idx_ty.unwrap(),
                    indices,
                ));
                break 'search;
            }
        }
        let Some((b, i, phi_id, cast, chain_ty, mut prefix, inbounds, idx_ty, indices)) = plan
        else {
            return changed;
        };
        changed = true;
        let idx_phi = super::id(header);
        let label_to_block: FxHashMap<Word, usize> = func
            .blocks
            .iter()
            .enumerate()
            .map(|(i, block)| (block.label_id().unwrap(), i))
            .collect();
        let mut operands = Vec::new();
        for (index, pred) in indices {
            let index = match index {
                IncomingIndex::At(x) => x,
                IncomingIndex::Same => idx_phi,
                IncomingIndex::Offset(offset) => {
                    let sum = super::id(header);
                    let add = Instruction::new(
                        Op::IAdd,
                        Some(idx_ty),
                        Some(sum),
                        vec![Operand::IdRef(idx_phi), Operand::IdRef(offset)],
                    );
                    insert_before_terminator(&mut func.blocks[label_to_block[&pred]], vec![add]);
                    sum
                }
            };
            operands.extend([Operand::IdRef(index), Operand::IdRef(pred)]);
        }
        let ac = super::id(header);
        prefix.push(Operand::IdRef(idx_phi));
        let opcode = if inbounds {
            Op::InBoundsAccessChain
        } else {
            Op::AccessChain
        };
        let mut after = vec![Instruction::new(opcode, Some(chain_ty), Some(ac), prefix)];
        let result = if let Some(cast_ty) = cast {
            let id = super::id(header);
            after.push(Instruction::new(
                Op::Bitcast,
                Some(cast_ty),
                Some(id),
                vec![Operand::IdRef(ac)],
            ));
            id
        } else {
            ac
        };
        let block = &mut func.blocks[b];
        block.instructions[i] = Instruction::new(Op::Phi, Some(idx_ty), Some(idx_phi), operands);
        let after_phis = block
            .instructions
            .iter()
            .take_while(|inst| inst.class.opcode == Op::Phi)
            .count();
        block.instructions.splice(after_phis..after_phis, after);
        replace_uses(func, phi_id, result);
    }
}

/// `value` as an [`Incoming`] of a phi of type `phi_ty` (a slice of `slice_elem`, or an element
/// pointer when `slice_elem` is `None`), if it is one of the forms `index_pointer_phis` handles.
fn describe_incoming(
    value: Word,
    phi_id: Word,
    slice_elem: Option<Word>,
    phi_ty: Word,
    defs: &FxHashMap<Word, Instruction>,
    types: &PtrTypes,
    type_of: &impl Fn(Word) -> Option<Word>,
) -> Option<Incoming> {
    let is_chain = |d: &&Instruction| {
        matches!(d.class.opcode, Op::AccessChain | Op::InBoundsAccessChain) && d.operands.len() >= 2
    };
    let element_chain = |id: Word| -> Option<Incoming> {
        let ac = defs.get(&id).filter(is_chain)?;
        let base = ac.operands[0].unwrap_id_ref();
        let last = ac.operands.last().unwrap().unwrap_id_ref();
        let (index, prefix) = if base == phi_id && ac.operands.len() == 2 {
            (IncomingIndex::Offset(last), Vec::new())
        } else {
            (
                IncomingIndex::At(last),
                ac.operands[..ac.operands.len() - 1].to_vec(),
            )
        };
        Some(Incoming {
            chain_ty: ac.result_type?,
            prefix,
            inbounds: ac.class.opcode == Op::InBoundsAccessChain,
            index,
        })
    };
    let Some(elem) = slice_elem else {
        return element_chain(value).filter(|d| d.chain_ty == phi_ty);
    };
    let elem_ptr = |id: Word| {
        type_of(id)
            .and_then(|ty| types.pointee.get(&ty))
            .is_some_and(|&(_, p)| p == elem)
    };
    if let Some(d) = defs.get(&value).filter(|d| d.class.opcode == Op::Bitcast) {
        let inner = d.operands[0].unwrap_id_ref();
        if elem_ptr(inner) {
            return element_chain(inner);
        }
        return whole(inner, elem, types, type_of);
    }
    whole(value, elem, types, type_of)
}

/// A pointer to a whole array or slice of `elem` as the access chain to its first element.
fn whole(
    array: Word,
    elem: Word,
    types: &PtrTypes,
    type_of: &impl Fn(Word) -> Option<Word>,
) -> Option<Incoming> {
    let &(class, pointee) = types.pointee.get(&type_of(array)?)?;
    if types.array_elem.get(&pointee) != Some(&elem) {
        return None;
    }
    Some(Incoming {
        chain_ty: *types.by_class_and_pointee.get(&(class, elem))?,
        prefix: vec![Operand::IdRef(array)],
        inbounds: true,
        index: IncomingIndex::At(types.zero_u32?),
    })
}
