"""A simple model for SN-injected magnetic fields
   Previous version made a grid of values for two IMFs (Salpeter, top-heavy)
   Assuming either a bimodal distribution of stellar magnetic fields
   or a signle Gaussian with only highly magnetized stars.
   In that version the magnetic field was considered constant=surface field.
   New in this version:
   -- added a dipole field configuration up to a core radius Rcore = 0.2*Rstar
   -- reduced the max magnetization of the OB stars from the absurd 10^5 to 10^3.
"""

import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

params = {'backend': 'wxAgg', 'lines.markersize' : 2, 'axes.labelsize': 18, 'font.size': 16, 'legend.fontsize': 16, 'xtick.labelsize': 16, 'ytick.labelsize': 16, 'axes.linewidth' : 2}
plt.rcParams.update(params)

def create_normal(Nmassive,mu,sigma):

   """Create a normal distribution of magnetic field values"""
   # Nmassive is the total number of massive stars

   bstars = np.random.normal(mu, sigma, int(Nmassive))
   return bstars

def create_bimodal(Nmassive,mu1,mu2,sigma1,sigma2,f1,f2):

   """Create a bimodal distribution of magnetic field values"""
   # Nmassive is the total number of massive stars
 
   bstars = np.concatenate((np.random.normal(mu1, sigma1, int(f1 * Nmassive)),
                    np.random.normal(mu2, sigma2, int(f2 * Nmassive))))#[:, np.newaxis]
   if int(f1 * Nmassive)+int(f2 * Nmassive) < Nmassive:
      bstars = np.concatenate((np.random.normal(mu1, sigma1, int(f1 * Nmassive)),
                    np.random.normal(mu2, sigma2, int(f2 * Nmassive)+1)))#[:, np.newaxis]
   if int(f1 * Nmassive)+int(f2 * Nmassive) < Nmassive-1:
      bstars = np.concatenate((np.random.normal(mu1, sigma1, int(f1 * Nmassive)+1),
                    np.random.normal(mu2, sigma2, int(f2 * Nmassive)+1)))#[:, np.newaxis]
   if int(f1 * Nmassive)+int(f2 * Nmassive) > Nmassive:
      bstars = np.concatenate((np.random.normal(mu1, sigma1, int(f1 * Nmassive)),
                    np.random.normal(mu2, sigma2, int(f2 * Nmassive)-1)))#[:, np.newaxis]

   # where bstars < 0, turn them to 0
   mask = bstars < 0.
   bstars[mask] = 0.

   return bstars

def nmassive_salpeter(Mmin_pop,Mstars,Mmin_sn,Mmax):

  # Salpeter IMF is xi(m) = m**-2.3
  nmassive = xi_0 *(Mmin_sn**-1.35 - Mmax**-1.35)/1.35
  return nmassive

def nmassive_topheavy(Mmin_pop,Mstars,Mmin_sn,Mmax):

  # Top-heavy part of the IMF is xi(m) = m**-1.5 (i.e. Marks, Kroupa et al. 2012)
  nmassive = xi_0 *(Mmin_sn**-0.5 - Mmax**-0.5)/0.5
  return nmassive

def powerlaw(a, b, g, size=1):
  """Power-law generator for pdf(x)\propto x^{g-1} for a<=x<=b"""
  r = np.random.random(size=size)
  ag, bg = a**g, b**g
  return (ag + (bg - ag)*r)**(1./g)

def dipole_from_surface(Bsur, Rstar):

  """ Calculate the B-field of a dipole knowing the surface field strength, Bsur. """
  # the minimum radius is either 0.2*Rstar or the radius at which B=1.e6
  limitfield = 1.e6 # maximum stable field (Augustson + 2016)
  Rcore1 = 0.2*Rstar
  Rmin   = np.cbrt(Bsur/limitfield)/Rstar
  # take as rmin whichever is larger
  if Rmin >= Rcore1:
     Rcore = Rmin
     Bcore = limitfield
  else:
     Rcore = Rcore1
     Bcore = Bsur*(Rstar/Rcore)**3. 

  #print('dipole:Bsur,Bcore',Bsur,Bcore)

  emag = (Bcore**2.*Rcore**3. + Bsur**2.*Rstar**6.*(1/Rcore**3.-1./Rstar**3.))/6. 
  #print('dipole:Bsur,Bcore,Rcore,Rstar,emag',Bsur,Bcore,Rcore,Rstar,emag)
  
  return emag

#### Main ####

#Mcl    = 1.e4    # mass of the parent cloud
#eff    = 1.e-1   # efficiency of star formation
#Mstars = Mcl*eff # total mass of formed stars
pc   = 3.08e18
Msun = 2.e33
rsun = 6.96e10 # cm
rsb  = 300.*pc
rsn  = 30.*pc
Volsb  = 4*np.pi*rsb**3./3.
Volsn  = 4*np.pi*rsn**3./3.

numclusters = 8
#masses      = np.linspace(3.,6.,num=numclusters)# total mass of formed stars
masses      = [3.,3.5,3.8,4.0,4.2,4.5,5.0,6.0]# total mass of formed stars
emag_mean = np.zeros(numclusters)
emag_min = np.zeros(numclusters)
emag_max = np.zeros(numclusters)
bfieldsb_mean = np.zeros(numclusters)
bfieldsb_min = np.zeros(numclusters)
bfieldsb_max = np.zeros(numclusters)
bfieldsn_mean = np.zeros(numclusters)
bfieldsn_min = np.zeros(numclusters)
bfieldsn_max = np.zeros(numclusters)
nsn      = np.zeros(numclusters)
#
#imf = 'salpeter' # 'salpeter' or 'topheavy'
bdis = 'gaussian'# 'bimodal' or 'gaussian'
bfstar = 'dipole' # 'dipole' or 'constant'

plot_bfields = True

imfs = ['salpeter','topheavy']
colorms = [['k','b'],['r','green']]
imfls = ['Salpeter', 'Top-heavy']
fig, ax = plt.subplots(figsize=(9,6))
fig1, ax1 = plt.subplots(figsize=(9,6))
for imf,colorm,imfl in zip(imfs,colorms,imfls):
   for i in range(numclusters):
      Mstars = 10.**masses[i]
      Mmin_sn = 8. # Minimum mass for SN in Msun
      #  Larson (2003) for the maximum stellar mass for this population
      Mmax_pop = 1.2*Mstars**0.45  # Maximum stellar mass for this population (empirical relation, see Weidner & Kroupa 2006)
      Mmin_pop = 0.01              # Minimum possible stellar mass 

      # How many stars are formed and how many of them have M* > Mmin_sn?
      if imf=='salpeter':
         xi_0     = Mstars / (Mmin_pop**-0.35 - Mmax_pop**-0.35)
         N_massive = int(nmassive_salpeter(Mmin_pop,Mstars,Mmin_sn,Mmax_pop))+1 
         # Create an array of stellar masses
         masses_stars = powerlaw(Mmin_sn, Mmax_pop, -1.35, size=N_massive)
         # if M*>300Msun the entire core collapses into a black hole and no heavy elements are ejected (Heger & Woosely 2002)
         #Mlimit = 300.
         #mask = masses_stars < Mlimit 
         #if len(masses_stars[mask]) > 0:
         #   masses_stars = masses_stars[mask]
         #   print(len(masses_stars))
         #   N_massive    = len(masses_stars)
         # Create radii array 
         # mass-radius relation for M*>1.66Msun (Demicran & Kahraman 1991) logR=a+blogM, with a=0.124 and b=0.555 
         radii_stars = 10.**(0.124 + 0.57*np.log10(masses_stars))
         radii_stars *= rsun
      if imf=='topheavy':
         xi_0     = Mstars / (Mmax_pop**0.5 - Mmin_pop**0.5)
         N_massive = int(nmassive_topheavy(Mmin_pop,Mstars,Mmin_sn,Mmax_pop))+1
         # Create an array of stellar masses
         masses_stars = powerlaw(Mmin_sn, Mmax_pop, -0.5, size=N_massive)
         # Create radii array 
         # mass-radius relation for M*>1.66Msun (Demicran & Kahraman 1991) logR=a+blogM, with a=0.124 and b=0.555 
         radii_stars = 10.**(0.124 + 0.57*np.log10(masses_stars))
         radii_stars *= rsun
      if imf=='delta':
         mstar = 20.
         N_massive = int(Mstars/mstar)+1 
         radii_stars = np.repeat(10.**(0.124 + 0.57*np.log10(mstar)),N_massive)
         radii_stars *= rsun

      #print('A cluster of ',Mstars,'solar masses produces',N_massive,' supernovae')
      nsn[i] = N_massive

      #print('min and max stellar radii',np.min(radii_stars),np.max(radii_stars))

      # Create grid of values:
      if bdis=='bimodal': 
         ngrid1 = 10; ngrid2 = 10; ngrid3 = 4
  
         m1min = 10; m1max = 100
         m2min = 500; m2max = 5000
         #m1min = 10; m1max = 500
         #m2min = 10000; m2max = 50000

         mus1 = np.linspace(m1min,m1max,num=ngrid1)
         sigmas1 =  0.1*mus1
         mus2 = np.linspace(m2min,m2max,num=ngrid2)
         sigmas2 =  0.1*mus2
         # 
         f1min = 0.8; f1max = 0.95

         fis1 = np.linspace(f1min,f1max,num=ngrid3)
 
         #s1 = 8; s2 = 1000
      if bdis=='gaussian':
         ngrid1 = 10; ngrid2 = 1; ngrid3 = 1

         m1min = 1000; m1max = 10000
         #m1min = 10000; m1max = 100000

         mus = np.linspace(m1min,m1max,num=ngrid1)
         # 
         sigmas =  0.1*mus #1000

      emag_total=np.zeros((ngrid3,ngrid1,ngrid2))
      bfield_sn=np.zeros((ngrid3,ngrid1,ngrid2))

      if plot_bfields:
         fig3, ax3 = plt.subplots(figsize=(8,6))
         ax3.set_xlabel(r'log ($B_{\rm *,sur}$/G)')
         ax3.set_ylabel(r'N (log ($B_{\rm *,sur}$/G))')
         if bdis=='bimodal':
            ax3.set_xlim([0.5,4.0])
         else:
            ax3.set_xlim([2.7,4.2])
         # For choosing colors from a colormap
         cmap = mpl.cm.get_cmap('tab20b')
         ncolors = ngrid1*ngrid2*ngrid3
         norm = mpl.colors.Normalize(vmin=0.0, vmax=float(ncolors))
         basename = 'cluster'+str(masses[i])

      mindex = 0 

      for f1 in range(ngrid3):
        print(f1)
        for m1 in range(ngrid1):
          for m2 in range(ngrid2):
             # Each star in masses_stars has a magnetic field from bfield_stars
             if bdis=='bimodal':
                bfield_stars = create_bimodal(N_massive,mus1[m1],mus2[m2],sigmas1[m1],sigmas2[m2],fis1[f1],1.-fis1[f1]) 
             if bdis=='gaussian':
                bfield_stars = create_normal(N_massive,mus[m1],sigmas[m1]) 
             if plot_bfields:
                #if (m1==0 or m1==ngrid1-1) and (m2==0 or m2==ngrid2-1) and (f1==0 or f1==ngrid3-1):
                colorVal = cmap(norm(float(mindex)))   
                #label = r'$\mu_1$='+str(mus1[m1])+r', $\mu_2$='+str(mus2[m2])+r', $f_2$='+'{0:.2f}'.format(1.-fis1[f1]) 
                mask = bfield_stars > 0
                y, binEdges = np.histogram(np.log10(bfield_stars[mask]), bins=50)
                bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
                ax3.plot(bincenters, y, '-', c=colorVal)#,label=label)

             # Magnetic energy per star:
             if bfstar=='constant':
                energies_stars = bfield_stars**2.*(4.*np.pi*(radii_stars)**3.)/3./8./np.pi
             if bfstar=='dipole':
                energies_stars = [(dipole_from_surface(Bst, Rst)) for (Bst, Rst) in zip(bfield_stars,radii_stars)] 
                energies_stars = np.array(energies_stars)
                #print(energies_stars.shape)
             # Magnetic field per supernova:
             bfields_sn = np.sqrt(energies_stars*8.*np.pi/Volsn)

             # Total magnetic energy "ejected" by these stars
             emag_total[f1,m1,m2] = np.sum(energies_stars)
             # Mean magnetic field over all the SNe
             bfield_sn[f1,m1,m2]  = np.mean(bfields_sn)
             mindex += 1

      if plot_bfields:
         #plt.legend()
         fig3.savefig('Bdistribution'+bdis+'_'+bfstar+basename+'.png')

      emag_mean[i] = np.mean(emag_total)
      emag_min[i]  = np.min(emag_total)
      emag_max[i]  = np.max(emag_total)
      # Assuming a superbubble
      bfieldsb_mean[i] = np.sqrt(emag_mean[i]*8.*np.pi/Volsb)
      bfieldsb_min[i] = np.sqrt(emag_min[i]*8.*np.pi/Volsb)
      bfieldsb_max[i] = np.sqrt(emag_max[i]*8.*np.pi/Volsb)
      # Assuming supernova remnants
      bfieldsn_mean[i] = np.mean(bfield_sn) 
      bfieldsn_min[i] = np.min(bfield_sn) 
      bfieldsn_max[i] = np.max(bfield_sn)
      print(i, emag_mean[i], emag_min[i], emag_max[i])
      print(i, bfieldsb_mean[i], bfieldsb_min[i], bfieldsb_max[i])
      print(i, bfieldsn_mean[i], bfieldsn_min[i], bfieldsn_max[i])

   ax.plot(masses,np.log10(emag_mean/(nsn*1.e51)),color=colorm[0],linestyle='-',label=r'$\langle E_{mag,tot}/E_{SN,tot} \rangle$, '+imfl)
   ax.fill_between(masses,np.log10(emag_min/(nsn*1.e51)),np.log10(emag_max/(nsn*1.e51)),color=colorm[0],alpha=0.2)
   ax1.plot(masses,np.log10(bfieldsb_mean/1.e-6),color=colorm[0],linestyle='-',label=r'$\langle B_{SB}\rangle$, '+imfl)
   ax1.plot(masses,np.log10(bfieldsn_mean/1.e-6),color=colorm[1],linestyle='--',label=r'$\langle B_{SN}\rangle$, '+imfl)
   ax1.fill_between(masses,np.log10(bfieldsb_min/1.e-6),np.log10(bfieldsb_max/1.e-6),color=colorm[0],alpha=0.2)
   ax1.fill_between(masses,np.log10(bfieldsn_min/1.e-6),np.log10(bfieldsn_max/1.e-6),color=colorm[1],alpha=0.2)

ax.set_xlabel(r'log $M_{cluster}/M_\odot$')
ax.set_ylabel(r'log $E_{mag,inj}/E_{SN,tot}$')
ax.legend()
fig.savefig('mass_emag_'+bdis+'_'+bfstar+'_twoimfs.png')

ax1.set_xlabel(r'log $M_{cluster}/M_\odot$')
ax1.set_ylabel(r'log $\langle B \rangle/\mu G$')
ax1.legend()
fig1.savefig('mass_bfield_'+bdis+'_'+bfstar+'_twoimfs.png')
