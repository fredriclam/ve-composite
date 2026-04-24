''' Composite viscoelasticity tools

Contents:
  * 

'''

import numpy as np
import matplotlib.pyplot as plt

def nu_KG(K, G):
  ''' Compute poisson ratio as function of K, G. '''
  nu = (3 * K - 2 * G) / (2 * (3 *K + G))
  return nu


class VoigtLaminate:

  @staticmethod
  def G(G1, G2, phi2):
    return (1.0 - phi2) * G1 + phi2 * G2
  
  @staticmethod
  def K(K1, K2, phi2):
    return (1.0 - phi2) * K1 + phi2 * K2


class ReussLaminate:

  @staticmethod
  def G(G1, G2, phi2):
    return 1.0 / ((1.0 - phi2) / G1 + phi2 / G2)

  @staticmethod
  def K(K1, K2, phi2):
    return 1.0 / ((1.0 - phi2) / K1 + phi2 / K2)


def G_Hashin_bounds(K1, K2, G1, G2, phi2) -> tuple:
  ''' Hashin bounds on composite shear modulus. These bounds apply for
  isotropic elastic materials. Inputs G1, G2, K1, K2, phi2 are arrays of the
  same size. Returns the lower and upper bounds on G as an np.array with new
  axis 0. '''

  def bound_qty(K2, G1, G2, phi1):
    phi2 = 1 - phi1
    return (G2 + phi1 / (1 / (G1-G2)
      + 6 * (K2 + 2 * G2) * phi2 / (5 * (3 * K2 + 4 * G2) * G2)))
  
  G_bounds = np.stack([bound_qty(K2, G1, G2, 1.0 - phi2),
                       bound_qty(K1, G2, G1, phi2)], axis=0)
  return np.array((G_bounds.min(axis=0), G_bounds.max(axis=0),))


class SphericalInclusion:

  @staticmethod
  def G_nu_G(nu1, G1, G2, phi2):
    ''' Spherical inclusion shear modulus for small volume fraction phi2
      Variant based on (nu, G) input.

      Reference:
        Christensen 1979, Chen and Lakes J. Mat. Sci. 

      Taylor expansion of G_Hashin lower bound, or from considering a single
      spherical (Eshelby) inclusion, subject to remote stresses at infinity.
    '''
    num = 15 * (1 - nu1) * (1 - G2 / G1) * (phi2)
    denom = 7 - 5 * nu1 + 2 * (4 - 5 * nu1 ) * G2 / G1
    return G1 * (1 - num/denom)

  @staticmethod
  def G(K1, G1, G2, phi2):
    ''' From Hashin 1983 Analysis of composites--a survey.
    
    '''
    denom = 9 * K1 + 8 * G1 + 6 * (K1 + 2*G1) * G2 / G1
    return G1 + (G2 - G1) * 5 * (3 * K1 + 4 * G1) * phi2 / denom

  @staticmethod
  def K(K1, K2, G1, phi2):
    ''' From Hashin 1983 Analysis of composites--a survey
    '''
    return K1 + (K2 - K1) * (3 * K1 + 4 * G1) / (3 * K2 + 4 * G1) * phi2
  
  @staticmethod
  def K_complex(K1, K2, G1, eta1, phi2, omega):
    ''' Spherical inclusion equivalent standard linear solid (SLS).
    Returns a complex K(omega), where omega is the input vector of frequencies.

    Equivalent to SphericalInclusion.K, evaluated using complex G1 for a Maxwell
    material. The complex G1(i*omega) for a Maxwell material is given by
    MaxwellMaterial.mod2complex. That is, this function is equivalent to
    
      SphericalInclusion.K(K1, K2, MaxwellMaterial.mod2complex(omega, G1), phi2)

    This formula is obtained by identifying the complex bulk modulus
               (s - z_0)    K1e + (K1e + K2e) / K2e * zetae s
      K(s) = A --------- =  ---------------------------------
               (s - p_0)          1 + zetae / K2e * s
    with the form of the SLS circuit model and identifying each of the component
    values K1e, K2e, zeta. 

                  ┌───K1e───┐   
                ──┤         ├── 
                  └─K2e──ζe─┘   
    
    The same complex output can be obtained by substituting directly into the
    pole-zero format.

    Input: Real (frequency)
    Output: Complex
    '''

    # Branch bulk moduli
    K1e = K1 * (1 + phi2 * (1 - K1/K2))
    K2e = (K2 - K1)*(K2 - K1) / K2 * phi2 * 4 * G1 / (3 * K2 + 4  * G1)    
    # Bulk viscosity function, dependent on crust shear viscosity eta 
    # For frequency domain, plug in (i * omega) == i * De * G / K 
    zetae = 4/3 * eta1 * (1 - K1/K2)**2 * phi2
    # Compute reused intermediate quantity (Deborah number for Maxwell branch)
    # Note omega == De*1j*G/K
    De = omega * zetae / K2e
    # Return complex valued K for SLS model in terms of components
    return (K1e + De * 1j * (K1e + K2e)) / (1 + De * 1j) 
  
  @staticmethod
  def K_pole_zero(K1, K2, G1, eta1, phi2):
    ''' Spherical inclusion equivalent standard linear solid (SLS) in pole
    zero form
               (s - z_0)
      K(s) = A ---------.
               (s - p_0)
    Returns (roots_num, roots_den, leading_coeff)

    Input: Real (frequency)
    Output: Real (zeros, poles, leading coefficient)
    '''

    # Branch bulk modului
    K1e = K1 * (1 + phi2 * (1 - K1/K2))
    K2e = (K2 - K1)*(K2 - K1) / K2 * phi2 * 4 * G1 / (3 * K2 + 4  * G1)    
    # Equivalent bulk viscosity
    zetae = 4/3 * eta1 * (1 - K1/K2)**2 * phi2

    # Array of zeros
    roots_num = np.array([
      -K1e * K2e / zetae / (K1e + K2e)
    ])
    # Array of poles (independent of phi2)
    roots_den = np.array([
      -K2e/zetae, # == -K2 / (4/3*eta2) * 4 * G1 / (3*K2 + 4*G1),
    ]) 
    leading_coeff = K1e + K2e
  
    return (roots_num, roots_den, leading_coeff)
  
  def G_pole_zero(K1, K2, G1, G2, eta1, eta2, phi2):
    ''' Spherical inclusion equivalent generalized Maxwell shear modulus in pole
    zero form

                s(s - z01)(s - z1)
    G(s) = A ------------------------
            (s - p0)(s - p1)(s - p2)

    Assumes a dilute spherical inclusion (phase 2) in a viscoelastic matrix
    (phase 1) with bulk elastic properties and deviatoric viscoelasticity.

    Returns (roots_num, roots_den, leading_coeff)

    Can be identified with the circuit

            ┌─G0e──η0e─┐ 
            │          │ 
           ─┼─G1e──η1e─┼─
            │          │ 
            └─G2e──η2e─┘ 

    Input: Real (frequency)
    Output: Real (zeros, poles, leading coefficient)
    '''

    roots_num = np.array([
      0,
      -((G1*((400*G1**2*G2**2*eta1**2*phi2**2 - 320*G1**2*G2**2*eta1**2*phi2 + 64*G1**2*G2**2*eta1**2 - 800*G1**2*G2**2*eta1*eta2*phi2**2 - 160*G1**2*G2**2*eta1*eta2*phi2 + 192*G1**2*G2**2*eta1*eta2 + 400*G1**2*G2**2*eta2**2*phi2**2 + 480*G1**2*G2**2*eta2**2*phi2 + 144*G1**2*G2**2*eta2**2 - 600*G1**2*G2*K1*eta1*eta2*phi2**2 + 600*G1**2*G2*K1*eta1*eta2*phi2 - 144*G1**2*G2*K1*eta1*eta2 + 600*G1**2*G2*K1*eta2**2*phi2**2 + 24*G1**2*G2*K1*eta2**2 + 225*G1**2*K1**2*eta2**2*phi2**2 - 270*G1**2*K1**2*eta2**2*phi2 + 81*G1**2*K1**2*eta2**2 + 600*G1*G2**2*K1*eta1**2*phi2**2 - 600*G1*G2**2*K1*eta1**2*phi2 + 144*G1*G2**2*K1*eta1**2 - 600*G1*G2**2*K1*eta1*eta2*phi2**2 - 24*G1*G2**2*K1*eta1*eta2 - 450*G1*G2*K1**2*eta1*eta2*phi2**2 + 540*G1*G2*K1**2*eta1*eta2*phi2 - 162*G1*G2*K1**2*eta1*eta2 + 225*G2**2*K1**2*eta1**2*phi2**2 - 270*G2**2*K1**2*eta1**2*phi2 + 81*G2**2*K1**2*eta1**2)**(1/2) + 9*G1*K1*eta2 - 15*G1*K1*eta2*phi2))/2 + (G1*(20*G1**2*phi2 - 8*G1**2 - 9*G1*K1 + 15*G1*K1*phi2)*(8*G1*eta1 + 12*G1*eta2 + 9*K1*eta1 + 12*K1*eta2 - 20*G1*eta1*phi2 + 20*G1*eta2*phi2 - 15*K1*eta1*phi2 + 30*K1*eta2*phi2))/(2*(12*G1 + 6*K1 + 20*G1*phi2 + 15*K1*phi2)))/(eta1*eta2*(8*G1**2 - 20*G1**2*phi2 + 12*G1*G2 + 9*G1*K1 + 6*G2*K1 + 20*G1*G2*phi2 - 15*G1*K1*phi2 + 15*G2*K1*phi2)) - (G1*(8*G1*eta1 + 12*G1*eta2 + 9*K1*eta1 + 12*K1*eta2 - 20*G1*eta1*phi2 + 20*G1*eta2*phi2 - 15*K1*eta1*phi2 + 30*K1*eta2*phi2))/(2*eta1*eta2*(12*G1 + 6*K1 + 20*G1*phi2 + 15*K1*phi2)),
      ((G1*((400*G1**2*G2**2*eta1**2*phi2**2 - 320*G1**2*G2**2*eta1**2*phi2 + 64*G1**2*G2**2*eta1**2 - 800*G1**2*G2**2*eta1*eta2*phi2**2 - 160*G1**2*G2**2*eta1*eta2*phi2 + 192*G1**2*G2**2*eta1*eta2 + 400*G1**2*G2**2*eta2**2*phi2**2 + 480*G1**2*G2**2*eta2**2*phi2 + 144*G1**2*G2**2*eta2**2 - 600*G1**2*G2*K1*eta1*eta2*phi2**2 + 600*G1**2*G2*K1*eta1*eta2*phi2 - 144*G1**2*G2*K1*eta1*eta2 + 600*G1**2*G2*K1*eta2**2*phi2**2 + 24*G1**2*G2*K1*eta2**2 + 225*G1**2*K1**2*eta2**2*phi2**2 - 270*G1**2*K1**2*eta2**2*phi2 + 81*G1**2*K1**2*eta2**2 + 600*G1*G2**2*K1*eta1**2*phi2**2 - 600*G1*G2**2*K1*eta1**2*phi2 + 144*G1*G2**2*K1*eta1**2 - 600*G1*G2**2*K1*eta1*eta2*phi2**2 - 24*G1*G2**2*K1*eta1*eta2 - 450*G1*G2*K1**2*eta1*eta2*phi2**2 + 540*G1*G2*K1**2*eta1*eta2*phi2 - 162*G1*G2*K1**2*eta1*eta2 + 225*G2**2*K1**2*eta1**2*phi2**2 - 270*G2**2*K1**2*eta1**2*phi2 + 81*G2**2*K1**2*eta1**2)**(1/2) - 9*G1*K1*eta2 + 15*G1*K1*eta2*phi2))/2 - (G1*(20*G1**2*phi2 - 8*G1**2 - 9*G1*K1 + 15*G1*K1*phi2)*(8*G1*eta1 + 12*G1*eta2 + 9*K1*eta1 + 12*K1*eta2 - 20*G1*eta1*phi2 + 20*G1*eta2*phi2 - 15*K1*eta1*phi2 + 30*K1*eta2*phi2))/(2*(12*G1 + 6*K1 + 20*G1*phi2 + 15*K1*phi2)))/(eta1*eta2*(8*G1**2 - 20*G1**2*phi2 + 12*G1*G2 + 9*G1*K1 + 6*G2*K1 + 20*G1*G2*phi2 - 15*G1*K1*phi2 + 15*G2*K1*phi2)) - (G1*(8*G1*eta1 + 12*G1*eta2 + 9*K1*eta1 + 12*K1*eta2 - 20*G1*eta1*phi2 + 20*G1*eta2*phi2 - 15*K1*eta1*phi2 + 30*K1*eta2*phi2))/(2*eta1*eta2*(12*G1 + 6*K1 + 20*G1*phi2 + 15*K1*phi2)),
    ])
    # Poles: independent of phi2
    roots_den = np.array([
      -G1/eta1,
      -((G1*((64*G1**2*G2**2*eta1**2 + 192*G1**2*G2**2*eta1*eta2 + 144*G1**2*G2**2*eta2**2 - 144*G1**2*G2*K1*eta1*eta2 + 24*G1**2*G2*K1*eta2**2 + 81*G1**2*K1**2*eta2**2 + 144*G1*G2**2*K1*eta1**2 - 24*G1*G2**2*K1*eta1*eta2 - 162*G1*G2*K1**2*eta1*eta2 + 81*G2**2*K1**2*eta1**2)**(1/2) + 9*G1*K1*eta2))/2 - (G1*(8*G1**2 + 9*K1*G1)*(8*G1*eta1 + 12*G1*eta2 + 9*K1*eta1 + 12*K1*eta2))/(2*(12*G1 + 6*K1)))/(eta1*eta2*(8*G1**2 + 12*G1*G2 + 9*G1*K1 + 6*G2*K1)) - (G1*(8*G1*eta1 + 12*G1*eta2 + 9*K1*eta1 + 12*K1*eta2))/(2*eta1*eta2*(12*G1 + 6*K1)),
      ((G1*((64*G1**2*G2**2*eta1**2 + 192*G1**2*G2**2*eta1*eta2 + 144*G1**2*G2**2*eta2**2 - 144*G1**2*G2*K1*eta1*eta2 + 24*G1**2*G2*K1*eta2**2 + 81*G1**2*K1**2*eta2**2 + 144*G1*G2**2*K1*eta1**2 - 24*G1*G2**2*K1*eta1*eta2 - 162*G1*G2*K1**2*eta1*eta2 + 81*G2**2*K1**2*eta1**2)**(1/2) - 9*G1*K1*eta2))/2 + (G1*(8*G1**2 + 9*K1*G1)*(8*G1*eta1 + 12*G1*eta2 + 9*K1*eta1 + 12*K1*eta2))/(2*(12*G1 + 6*K1)))/(eta1*eta2*(8*G1**2 + 12*G1*G2 + 9*G1*K1 + 6*G2*K1)) - (G1*(8*G1*eta1 + 12*G1*eta2 + 9*K1*eta1 + 12*K1*eta2))/(2*eta1*eta2*(12*G1 + 6*K1)),
    ]) 
    leading_coeff = ((G1*(12*G1 + 6*K1 + 20*G1*phi2 + 15*K1*phi2))/(6*(2*G1 + K1))
                    - (25*G1**2*phi2*(4*G1 + 3*K1)**2)/(6*(2*G1 + K1)*(G2*(12*G1 + 6*K1) + 8*G1**2 + 9*G1*K1)))
  
    return (roots_num, roots_den, leading_coeff)


class PlateletInclusion:

  @staticmethod
  def G(K2, G1, G2, phi2):
    ''' Dilute platelet inclusion shear modulus for small volume fraction phi2
      Christensen 1979, Chen and Lakes J. Mat. Sci.  Platelets are randomly
      oriented.
    '''
    _f1 = (9*K2 + 4*(G1 + 2*G2)) / (K2 + (4/3)*G2) + 6 * G1 / G2
    return G1 + phi2 * (G2 - G1) / 15 * _f1
  
  @staticmethod
  def K(K1, G2, K2, phi2):
    ''' Bulk modulus. '''
    return K1 + (K2 - K1) * (3 * K1 + 4 * G2) / (3 * K2 + 4 * G2) * phi2


class MaxwellMaterial:

  @staticmethod
  def mod2complex(De, G):
    ''' Complex modulus G(omega) as a function of Deborah number. Here Deborah
    number is defined as
      De == freq * viscosity / modulus
    using the viscosity of this MaxwellMaterial. '''
    return G * (De*De + De * 1j) / (De*De + 1)

  @staticmethod
  def loss_tangent(z):
    return np.imag(z) / np.real(z)


def pole_zero_split(num:np.array, den:np.array) -> np.array:
  '''
  Splits a pole-zero composite fraction into independent terms.
  A pole-zero fraction of the form

               prod_j (s - z_j)                        B_i
  G(s) = G_∞ -------------------- = G_∞ ( 1 + sum_i --------- )
               prod_j (s - p_j)                      s - p_i 
               
  Returns vector of coefficients B_i, where

             prod_j (p_i - z_j)
   B_i = ---------------------------
          prod_{j != i} (p_i - p_j)

  are the partial fraction coefficients. 
  '''
  
  # Compute (p_i - p_j) matrix
  diff_p = (den[:,np.newaxis] - den[np.newaxis,:])
  # Overwrite diagonals (p_i - p_j) with ones
  diff_p[np.arange(den.size), np.arange(den.size)] = 1
  # Return B_i (units of frequency)
  return (den[:,np.newaxis] - num).prod(axis=1) / diff_p.prod(axis=1)