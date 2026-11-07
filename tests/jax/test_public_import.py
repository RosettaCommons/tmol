"""Public discovery does not initialize an array framework or native runtime."""

import subprocess
import sys
import textwrap


def test_public_import_is_lazy_and_resolves_existing_modules():
    subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""
                import sys
                import tmol_jax

                assert 'jax' not in sys.modules
                assert 'torch' not in sys.modules
                assert 'tmol_jax_native' not in sys.modules
                assert set(tmol_jax.__all__) <= set(dir(tmol_jax))
                from tmol_jax import ScoreType, PoseStack, io
                from tmol_jax.score import ScoreType as direct_score_type
                from tmol_jax.pose import PoseStack as direct_pose_stack

                assert ScoreType is direct_score_type
                assert PoseStack is direct_pose_stack
                assert io is sys.modules['tmol_jax.io']
                assert tmol_jax.PoseStack is PoseStack
                assert 'torch' not in sys.modules
                try:
                    tmol_jax.no_such_public_api
                except AttributeError:
                    pass
                else:
                    raise AssertionError('unknown public name must raise')
                """),
        ],
        check=True,
    )
