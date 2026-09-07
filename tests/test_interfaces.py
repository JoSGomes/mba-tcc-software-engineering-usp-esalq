"""Verify that seart_ghs and woc expose identical fetch_repositories signatures."""

import inspect
import pytest


def test_fetch_repositories_same_signature():
    from extracao import seart_ghs, woc

    seart_sig = inspect.signature(seart_ghs.fetch_repositories)
    woc_sig = inspect.signature(woc.fetch_repositories)

    assert set(seart_sig.parameters) == set(woc_sig.parameters), (
        f"Parameter names differ:\n  seart_ghs: {list(seart_sig.parameters)}\n"
        f"  woc:       {list(woc_sig.parameters)}"
    )

    # cache_dir intentionally has different defaults (source-specific cache paths)
    skip_default_check = {"cache_dir"}

    for name in seart_sig.parameters:
        sp = seart_sig.parameters[name]
        wp = woc_sig.parameters[name]
        assert sp.annotation == wp.annotation, (
            f"Annotation mismatch for '{name}': seart={sp.annotation!r}, woc={wp.annotation!r}"
        )
        if name not in skip_default_check:
            assert sp.default == wp.default or (
                sp.default is inspect.Parameter.empty and wp.default is inspect.Parameter.empty
            ), f"Default mismatch for '{name}': seart={sp.default!r}, woc={wp.default!r}"


def test_fetch_repositories_return_type():
    """Both functions must be generators (return Iterator[RepoRecord])."""
    import types
    from extracao import seart_ghs, woc

    # woc raises NotImplementedError but must still be a generator function
    assert inspect.isgeneratorfunction(woc.fetch_repositories), (
        "woc.fetch_repositories must be a generator function"
    )
    # seart_ghs is also a generator
    assert inspect.isgeneratorfunction(seart_ghs.fetch_repositories), (
        "seart_ghs.fetch_repositories must be a generator function"
    )


def test_woc_raises_not_implemented():
    from extracao import woc

    gen = woc.fetch_repositories()
    with pytest.raises(NotImplementedError):
        next(gen)


def test_repo_record_fields():
    """RepoRecord canonical fields are all present."""
    from extracao.seart_ghs import RepoRecord
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(RepoRecord)}
    required = {
        "repo_id", "name", "description", "language",
        "stars", "forks", "watchers", "open_issues", "size_kb",
        "has_readme", "has_license", "has_wiki",
        "commits", "contributors", "created_at", "updated_at",
        "readme_text", "collected_at",
    }
    assert required.issubset(field_names), (
        f"Missing fields in RepoRecord: {required - field_names}"
    )


def test_seeds_module():
    """config.seeds sets GLOBAL_SEED and set_global_seeds is callable."""
    from config import seeds

    assert hasattr(seeds, "GLOBAL_SEED")
    assert isinstance(seeds.GLOBAL_SEED, int)
    assert callable(seeds.set_global_seeds)
    seeds.set_global_seeds(0)  # must not raise
