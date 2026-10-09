"""Numerical checks for the research geometry, independent of text benchmarks."""
import unittest
import numpy as np
from curved_head import Head, ball, distance, probabilities


class GeometryTests(unittest.TestCase):
    def test_all_head_gradients_against_central_differences(self):
        rng = np.random.default_rng(13)
        x = rng.normal(size=(3,4))
        y = np.array([0,1,2])
        for kind,c in [("dot",0), ("flat",0), ("curved",.1), ("curved",1), ("blend",1)]:
            h = Head(4,2,3,7,kind,c,np.ones(3))
            _, grads = h.loss_grad(x,y)
            for key, values in h.params.items():
                for index in np.ndindex(values.shape):
                    old, eps = values[index], 1e-6
                    values[index] = old+eps
                    positive = h.loss_grad(x,y)[0]
                    values[index] = old-eps
                    negative = h.loss_grad(x,y)[0]
                    values[index] = old
                    numerical = (positive-negative)/(2*eps)
                    self.assertAlmostEqual(grads[key][index],numerical,places=6,msg=f"{kind}/{c}/{key}/{index}")

    def test_flat_distance_is_dot_softmax_with_adjusted_bias(self):
        q = np.array([[.3,.4],[-.2,.5]])
        p = np.array([[.1,.3],[.8,-.2],[-.4,.7]])
        bias = np.array([.1,-.5,.8])
        distances,_ = distance(q,p,0)
        a = probabilities(bias-distances)[0]
        b = probabilities(2*q@p.T+bias-np.sum(p*p,axis=1))[0]
        np.testing.assert_allclose(a,b,atol=1e-14)

    def test_curved_distance_flat_limit_symmetry_and_coincidence(self):
        raw = np.array([[.3,.4],[-.2,.5],[0.,0.]])
        q,_ = ball(raw,1.)
        d,_ = distance(q,q,1.)
        np.testing.assert_allclose(d,d.T,atol=1e-14)
        np.testing.assert_allclose(np.diag(d),0,atol=1e-14)
        q,_ = ball(raw,1e-8)
        curved,_ = distance(q,q,1e-8)
        flat,_ = distance(raw,raw,0)
        np.testing.assert_allclose(curved,flat,atol=1e-8)
        # Formula is geodesic squared distance, not coordinate distance.
        radial,_ = distance(np.zeros((1,2)),np.array([[.5,0.]]),1.)
        self.assertAlmostEqual(radial[0,0],np.arctanh(.5)**2,places=14)

    def test_valid_probability_mass_and_boundary_rejection(self):
        probs,_ = probabilities(np.array([[1000.,1001.,-1000.]]))
        self.assertAlmostEqual(float(probs.sum()),1.,places=14)
        self.assertTrue(np.all(probs>=0))
        with self.assertRaises(FloatingPointError):
            distance(np.array([[1.,0.]]),np.zeros((1,2)),1.)
        # Squared distance has a finite derivative at coincident points.
        h = Head(2,2,3,1,"curved",1.,np.ones(3))
        h.params["projection"].fill(0)
        h.params["tokens"].fill(0)
        loss, grads = h.loss_grad(np.ones((2,2)),np.array([1,2]))
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(all(np.all(np.isfinite(g)) for g in grads.values()))


if __name__ == "__main__":
    unittest.main()
