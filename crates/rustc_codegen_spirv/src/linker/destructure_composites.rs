//! Simplify `OpCompositeExtract` pointing to `OpCompositeConstruct`s / `OpCompositeInsert`s.
//! Such constructions arise after inlining, when using multi-argument closures
//! (and other `Fn*` trait implementations). These composites can frequently be invalid,
//! containing pointers, `OpFunctionArgument`s, etc. After simplification, components
//! will become valid targets for `OpLoad`/`OpStore`.
use super::apply_rewrite_rules;
use rspirv::dr::{Function, Instruction};
use rspirv::spirv::{Op, Word};
use rustc_data_structures::fx::FxHashMap;

pub fn destructure_composites(function: &mut Function) {
    let mut rewrite_rules = FxHashMap::default();
    let reference: FxHashMap<_, _> = function
        .all_inst_iter()
        .filter_map(|inst| match inst.class.opcode {
            Op::CompositeConstruct => Some((inst.result_id.unwrap(), inst.clone())),
            Op::CompositeInsert if inst.operands.len() >= 3 => {
                Some((inst.result_id.unwrap(), inst.clone()))
            }
            _ => None,
        })
        .collect();

    for inst in function.all_inst_iter_mut() {
        if inst.class.opcode == Op::CompositeExtract && inst.operands.len() >= 2 {
            let path: Vec<u32> = inst.operands[1..]
                .iter()
                .map(|op| op.unwrap_literal_bit32())
                .collect();
            if let Some(origin_id) =
                resolve_extract(&reference, inst.operands[0].unwrap_id_ref(), &path)
            {
                rewrite_rules.insert(
                    inst.result_id.unwrap(),
                    rewrite_rules.get(&origin_id).map_or(origin_id, |id| *id),
                );
                *inst = Instruction::new(Op::Nop, None, None, vec![]);
            }
        }
    }

    // Transitive closure computation
    let mut closed_rewrite_rules = rewrite_rules.clone();
    for value in closed_rewrite_rules.values_mut() {
        while let Some(next) = rewrite_rules.get(value) {
            *value = *next;
        }
    }

    // Remove instructions replaced by NOPs, as well as unused composite values.
    for block in function.blocks.iter_mut() {
        block
            .instructions
            .retain(|inst| inst.class.opcode != Op::Nop);
    }
    apply_rewrite_rules(&closed_rewrite_rules, &mut function.blocks);
}

/// The value at `path` inside `composite`, if the `OpCompositeConstruct`s and
/// `OpCompositeInsert`s (at any depth) that built it name it directly. An insert at
/// exactly `path` gives its object; one at a prefix of `path` continues inside its
/// object; one off `path` continues into the composite it was inserted into. An
/// insert below `path` leaves no existing value equal to the extraction.
fn resolve_extract(
    reference: &FxHashMap<Word, Instruction>,
    mut composite: Word,
    mut path: &[u32],
) -> Option<Word> {
    loop {
        if path.is_empty() {
            return Some(composite);
        }
        let inst = reference.get(&composite)?;
        match inst.class.opcode {
            Op::CompositeConstruct => {
                composite = inst.operands.get(path[0] as usize)?.unwrap_id_ref();
                path = &path[1..];
            }
            Op::CompositeInsert => {
                let insert_path: Vec<u32> = inst.operands[2..]
                    .iter()
                    .map(|op| op.unwrap_literal_bit32())
                    .collect();
                if path.starts_with(&insert_path) {
                    composite = inst.operands[0].unwrap_id_ref();
                    path = &path[insert_path.len()..];
                } else if insert_path.starts_with(path) {
                    return None;
                } else {
                    composite = inst.operands[1].unwrap_id_ref();
                }
            }
            _ => return None,
        }
    }
}
