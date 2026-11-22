"""Phase 3 tests (offline): variant generation, selection and lineage.

The structural rule under test: selection goes through a scorer based on the
suite (core/fitness/) and leaves an append-only lineage of EVERYTHING evaluated.
"""

import json
import random

from evolution.selection import select
from evolution.variants import Variant, generate_variants, genome_to_overrides


def _padre(agent="reviewer"):
    return Variant(variant_id=f"{agent}-g0-actual", agent_id=agent,
                   generation=0, parent_id=None, genome={})


def test_generate_variants_produce_hijos_unicos_y_con_linaje():
    padre = _padre()
    hijos = generate_variants(padre, n=4, rng=random.Random(7),
                              model_pool=["small", "tiny"])
    assert len(hijos) == 4
    genomas = {json.dumps(h.genome, sort_keys=True) for h in hijos}
    assert len(genomas) == 4                        # no duplicates
    for h in hijos:
        assert h.parent_id == padre.variant_id      # lineage linked
        assert h.generation == 1
        assert h.mutation != "origen"               # records what changed


def test_ids_unicos_entre_generaciones_con_contador():
    import itertools
    padre = _padre()
    counter = itertools.count()
    seen = set()
    g1 = generate_variants(padre, n=3, rng=random.Random(1), model_pool=["small", "tiny"],
                           seen=seen, id_counter=counter)
    # Dry generation: the same parent is mutated again.
    g2 = generate_variants(padre, n=3, rng=random.Random(2), model_pool=["small", "tiny"],
                           seen=seen, id_counter=counter)
    ids = [v.variant_id for v in g1 + g2]
    assert len(ids) == len(set(ids))                # NO repeated id
    # `seen` persists: the g2 genomes do not repeat those of g1.
    genomas = [json.dumps(v.genome, sort_keys=True) for v in g1 + g2]
    assert len(genomas) == len(set(genomas))


def test_mutate_no_genera_noop_contra_valor_efectivo():
    """With an empty genome but effective model 'small', swap_model must not pick
    'small' (it would be a no-op that wins through fitness noise)."""
    import random as _r
    padre = _padre("navigator")
    efectivo = {"model": "small", "temperature": 0.2, "num_predict": 1024}
    # Many samples: none may set model='small', temp=0.2 or np=1024.
    for s in range(40):
        hijos = generate_variants(padre, n=3, rng=_r.Random(s),
                                  model_pool=["small", "tiny"], effective=efectivo)
        for h in hijos:
            assert h.genome.get("model") != "small"
            assert h.genome.get("gen_options", {}).get("temperature") != 0.2
            assert h.genome.get("gen_options", {}).get("num_predict") != 1024


def test_genome_to_overrides_construye_prompt_y_opciones():
    ov = genome_to_overrides(
        {"directives": ["Máximo tres frases."], "gen_options": {"temperature": 0.4},
         "model": "tiny"},
        base_prompt="BASE",
    )
    assert ov["system_prompt"].startswith("BASE")
    assert "Máximo tres frases." in ov["system_prompt"]
    assert ov["gen_options"] == {"temperature": 0.4}
    assert ov["model"] == "tiny"
    assert genome_to_overrides({}, "BASE") == {}    # empty genome = current config


def test_select_puntua_ordena_y_escribe_linaje(tmp_path):
    padre = _padre()
    hijos = generate_variants(padre, n=3, rng=random.Random(3),
                              model_pool=["small", "tiny"])

    # Injected scorer (in production: make_suite_scorer -> core/fitness/).
    # It rewards having more directives; deterministic and trivial to verify.
    def scorer(v):
        score = 0.5 + 0.1 * len(v.genome.get("directives", []))
        return {"mean": score, "results": [{"name": "t", "score": score}]}

    linaje = tmp_path / "reviewer.jsonl"
    mejores = select(hijos, scorer, keep=2, lineage_path=linaje)

    assert len(mejores) == 2
    assert mejores[0].score >= mejores[1].score
    filas = [json.loads(l) for l in linaje.read_text(encoding="utf-8").splitlines()]
    assert len(filas) == 3                          # EVERYTHING evaluated is recorded
    assert {f["variant_id"] for f in filas} == {h.variant_id for h in hijos}
    assert all(f["genome"] is not None and f["score"] is not None for f in filas)


def test_select_respeta_el_tope_de_tiempo(tmp_path):
    import time
    hijos = generate_variants(_padre(), n=3, rng=random.Random(5),
                              model_pool=["small", "tiny"])
    linaje = tmp_path / "lin.jsonl"
    # Deadline in the past: NOTHING must be evaluated.
    out = select(hijos, scorer=lambda v: {"mean": 1.0, "results": []},
                 keep=1, lineage_path=linaje, deadline=time.time() - 1)
    assert out == []                                # nothing evaluated, nothing to return
    filas = [json.loads(l) for l in linaje.read_text(encoding="utf-8").splitlines()]
    assert all(f.get("skipped") for f in filas)     # but the lineage records why
