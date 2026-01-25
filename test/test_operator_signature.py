from tinyphysics.operators.spatial import operator_signature


def test_operator_signature():
  sig = operator_signature(["poisson_solve2", "grad2", "grad2"])
  assert sig == "poisson_solve2->grad2->grad2"
