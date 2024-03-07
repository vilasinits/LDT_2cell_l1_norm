import unittest
from cosmology_module import Cosmology
import sys

class TestCosmology(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Common parameters for all test methods
        cls.H0 = float(sys.argv[1]) if len(sys.argv) > 1 else 100.
        cls.Ob = float(sys.argv[2]) if len(sys.argv) > 2 else 0.046
        cls.Oc = float(sys.argv[3]) if len(sys.argv) > 3 else 0.233
        cls.mnu = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
        cls.ns = float(sys.argv[5]) if len(sys.argv) > 5 else 0.97
        cls.As_ = float(sys.argv[6]) if len(sys.argv) > 6 else 2.1
        cls.Omnu = cls.mnu / 93.14 / cls.H0 / cls.H0
        cls.Om = cls.Ob + cls.Oc + (cls.mnu / 93.14 / cls.H0 / cls.H0)
        cls.As = cls.As_ * 1e-9
        cls.speed_light = 299792.458
        cls.Ol = 1.0 - cls.Om
        

    def test_initialization(self):
        # Create an instance of Cosmology with placeholder zs
        cosmology_instance = Cosmology(self.H0, self.Ob, self.Oc, self.mnu, self.ns, self.As_, zs=None)

        # Assert statements to check if initialization is correct
        self.assertEqual(cosmology_instance.h, self.H0 / 100.)
        self.assertEqual(cosmology_instance.Om, self.Ob + self.Oc + self.Omnu)
        self.assertEqual(cosmology_instance.Oc, self.Om - self.Ob - self.Omnu)
        self.assertEqual(1, self.Om + self.Ol)

    def test_chi_and_redshift_conversion(self):
        # Create an instance of Cosmology with a specific zs value
        cosmology_instance = Cosmology(self.H0, self.Ob, self.Oc, self.mnu, self.ns, self.As_, zs=1.0)
        # Test the conversion functions
        chi_value = cosmology_instance.get_chi(0.5)
        self.assertAlmostEqual(cosmology_instance.get_z_from_chi(chi_value), 0.5, places=5)

    def test_lensing_weight(self):
        # Create an instance of Cosmology with a specific zs value
        cosmology_instance = Cosmology(self.H0, self.Ob, self.Oc, self.mnu, self.ns, self.As_, zs=1.0)
        # Test the lensing weight function
        chi_value = cosmology_instance.get_chi(0.5)
        chi_source = cosmology_instance.get_chi(cosmology_instance.zsource)
        lensing_weight = cosmology_instance.get_lensing_weight(chi_value, chi_source)
        # Define the expected result based on the lensing weight formula
        expected_result = 1.5 * cosmology_instance.Om * (cosmology_instance.speed_light ** -2.) * (
                (cosmology_instance.H0) ** 2.) * chi_value * (1 - (chi_value / chi_source)) * (
                                  1 + cosmology_instance.get_z_from_chi(chi_value))
        # Use assertAlmostEqual for floating-point comparisons
        self.assertAlmostEqual(lensing_weight, expected_result, places=5)

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
