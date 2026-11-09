# AtomWorks reference fixture

`ubq_atomworks.pt` is committed reference data used by the Atom37 input tests.
It contains `coords`, `residue_type`, `chain_iid`, and `pose_stack_coords`.

The source is `../pdb/1ubq.pdb`, loaded on CPU with `no_optH=True` and
`missing_density_distance_threshold=0.0`, then exported with
`atomworks_from_pose_stack`. The original one-time generator is preserved at
commit `75d920a9d69896d84b750fd4c5943d5fe4565fb9` under
`tmol/tests/data/atomworks/_generate_fixture.py`.

Tests read these fixed values; they must not regenerate them from the
implementation being tested. An intentional reference update should document
its source revision and scientific justification.
