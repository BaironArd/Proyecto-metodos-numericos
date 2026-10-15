import unittest

import numpy as np

from interfaz_grafica import formatear_detalle_bairstow, resolver_bairstow


class BairstowTests(unittest.TestCase):
    def assert_roots_reconstruct_polynomial(self, coefficients, initial_r=1, initial_s=-1):
        roots, history, factors = resolver_bairstow(
            coefficients,
            iteraciones_maximas=100,
            tolerancia=1e-10,
            r_inicial=initial_r,
            s_inicial=initial_s,
        )

        self.assertEqual(len(roots), len(coefficients) - 1)
        np.testing.assert_allclose(np.poly(roots) * coefficients[0], coefficients, atol=1e-8)
        self.assertTrue(factors)
        return history

    def test_quartic_with_real_and_complex_roots(self):
        roots, history, _ = resolver_bairstow([1, -3, 3, -3, 2], 100, 1e-10, 1, -1)
        np.testing.assert_allclose(np.poly(roots), [1, -3, 3, -3, 2], atol=1e-8)
        self.assertTrue(history)
        first_step = history[0]
        self.assertEqual(len(first_step["b"]), 5)
        self.assertAlmostEqual(first_step["norma"], max(
            abs(first_step["residuo_x"]), abs(first_step["residuo_constante"])
        ))
        np.testing.assert_allclose(
            first_step["jacobiano"] @ first_step["correcciones"],
            [-first_step["residuo_x"], -first_step["residuo_constante"]],
            atol=1e-10,
        )
        self.assertLessEqual(history[-1]["norma"], 1e-10)

    def test_iteration_explanation_contains_substitutions_and_deflation(self):
        _, history, _ = resolver_bairstow([1, -3, 3, -3, 2], 100, 1e-10, 1, -1)
        explanation = formatear_detalle_bairstow(history[1], 1e-10)
        converged_explanation = formatear_detalle_bairstow(history[-1], 1e-10)

        self.assertIn("b2 = a2 + r·b1 + s·b0", explanation)
        self.assertIn("c2 =", explanation)
        self.assertIn("d2 =", explanation)
        self.assertIn("J numérica", explanation)
        self.assertIn("r siguiente", explanation)
        self.assertIn("Deflación:", converged_explanation)

    def test_preserves_nonmonic_leading_factor(self):
        roots, _, factors = resolver_bairstow([2, -6, 6, -6, 4], 100, 1e-10, 1, -1)
        np.testing.assert_allclose(np.poly(roots) * 2, [2, -6, 6, -6, 4], atol=1e-8)
        self.assertEqual(factors[0], '2')

    def test_cubic_polynomial(self):
        history = self.assert_roots_reconstruct_polynomial([1, -6, 11, -6])
        self.assertTrue(history)

    def test_quadratic_polynomial(self):
        history = self.assert_roots_reconstruct_polynomial([1, 0, 1], initial_r=0, initial_s=-1)
        self.assertEqual(history, [])

    def test_rejects_zero_leading_coefficient(self):
        with self.assertRaisesRegex(ValueError, "principal"):
            resolver_bairstow([0, 1, 2], 10, 1e-8, 1, -1)

    def test_rejects_nonpositive_tolerance(self):
        with self.assertRaisesRegex(ValueError, "tolerancia"):
            resolver_bairstow([1, 0, 1], 10, 0, 1, -1)


if __name__ == "__main__":
    unittest.main()