'''This module was intially created by Zhilei Xu (zhileixu@space.mit.edu) and modified by Michael Brewer 
for the CLASS experiment. 
Zhilei adapted the CLASS-specific code for general use. If you use this package please cite the 
related paper (Zhilei Xu et al 2020 ApJ 891 134).
'''

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import curve_fit
from datetime import datetime, timedelta
import cPickle as pickle
import ephem

SIG2FWHM = 2.35482
MEAN_RADIUS = 1737.4 #km
REF_ANG_RADIUS = 948.0 / 3600.0 #degrees
AU =  149597870.7 #km
MEAN_DISK_BT = {'q_band' : 221.04,
                'w1_band': 218.04,
                'gl_band': 215.94,
                'gh_band': 213.65} 

# A class to hold data
class MoonSimData:
    def __init__(self, recv='q_band'):
        #Moon brightness temperature map (degrees K)
        self.moon_temp = None
        #Moon polarization map
        self.moon_u  = None
        #Moon illumination map (percent illuminated)
        self.moon_illum = None
        #Scaled map x coordinate grid (degrees)
        self.x = None
        #Scaled map y coordinate grid (degrees)
        self.y = None
        #Scaled map size (degrees)
        self.map_size = None
        #Scaled map resolution (degrees)
        self.res = None
        #Moon phase angle (radians)
        self.phase = None
        #Moon sub-solar longitude (radians)
        self.sslong = None 
        #Position angle of North Lunar Pole (wrt NCP, positive CCW on the sky, radians)
        self.pa_p = None 
        #Position angle of bright limb (wrt NCP, positive CCW on the sky, radians)
        self.pa_bl = None 
        #Moon parallactic angle (positive CCW on the sky, radians)
        self.par_ang = None 
        #Moon distance (topocentric km)
        self.distance = None
        #Moon radius (topocentric degrees)
        self.radius = None
        #Moon illumination fraction (positive waxing, negative waning)
        self.illum_frac = None
        #Mean disk averaged brightness temperature (non-convolved)
        self.mean_bt = MEAN_DISK_BT[recv]
        #Standard reference radius (topocentric degrees)
        self.ref_radius = REF_ANG_RADIUS

def gauss_map_gen(moon_x, moon_y, fwhm_x=1.5, fwhm_y=1.5, theta=0.0):
    '''Generate N x N elliptical Gaussian map given a set of x, y coordinates such
       as those returned by moon_sim_map
    '''
    theta_rad = np.radians(theta)
    sigma_x, sigma_y = fwhm_x/SIG2FWHM, fwhm_y/SIG2FWHM
    beam_solid_angle = 2.0 * np.pi * sigma_x * sigma_y
    x_p =  moon_x * np.cos(theta_rad) + moon_y * np.sin(theta_rad)
    y_p = -moon_x * np.sin(theta_rad) + moon_y * np.cos(theta_rad)
    beam = np.exp(-0.5 * ((x_p/sigma_x)**2 + (y_p/sigma_y)**2))               
    return beam, beam_solid_angle

class moon_sim:

    def __init__(self, dt=None, recv='q_band', fwhm=1.5, res=0.01, map_size=0.6):
        #Datetime for Moon simulation
        self.dt = dt
        #Receiver
        self.recv = recv
        #FWHM for convolution
        self.fwhm = fwhm
        #Map resolution (degrees, will be scaled to keep image size constant)
        self.res = res
        #Map size (degrees, will be scaled to keep image size constant)
        self.map_size = map_size
        self.map_data = MoonSimData(recv)
        if dt is not None:
            #Ephemeris Setup
            self.ephem = CLASSEphem()
            self.ephem.site.pressure = 0
            self.ephem.site.date = dt
            self.moon = self.ephem.get_object('Moon')
            self.sun  = self.ephem.get_object('Sun')
            self.map_data.distance = self.moon.earth_distance * AU
            self.map_data.radius = np.degrees(MEAN_RADIUS / self.map_data.distance)

    def constants(self):
        return self.map_data.mean_bt, self.map_data.ref_radius

    def moon_center_temps(self, fwhm_x=1.5, fwhm_y=1.5, theta=0.0):
        '''
        Convolve an elliptical Gaussian beam with the moon center and return the resulting
        beam diluted peak antenna and effective brightness temperatures.
        The provided beam parameters may be either a single instance or a list.
        Useful for looking at temperatures as a function of beamwidth or for a set of 
        detector beam parameters.
        '''
        if self.map_data.moon_temp is None: self.moon_sim_map(convolve=False)

        moon_solid_angle = np.pi * self.map_data.radius**2
        #moon_sim_map scales resolution
        resolution = self.map_data.res
        mask = np.where(self.map_data.moon_temp > 10.)
        if isinstance(fwhm_x, list):
            cent_at, cent_bt = [],[]
            for i in range(len(fwhm_x)):
                beam, beam_solid_angle = gauss_map_gen(self.map_data.x[mask], self.map_data.y[mask], fwhm_x[i], fwhm_y[i], theta[i])
                # Convolve beam at moon center
                moon_beam  = np.sum(self.map_data.moon_temp[mask] * beam) * resolution**2
                # Moon center effective brightness temperature
                #cent_bt.append(moon_beam /  min(moon_solid_angle, beam_solid_angle))
                # Moon center brightness temperature
                cent_bt.append(moon_beam / (np.sum(beam) * resolution**2))
                # Moon center beam convolved  antenna temperature
                cent_at = moon_beam / beam_solid_angle
            return cent_at, cent_bt
        else:
            beam, beam_solid_angle = gauss_map_gen(self.map_data.x[mask], self.map_data.y[mask], fwhm_x, fwhm_y, theta)
            # Convolve beam at moon center
            moon_beam  = np.sum(self.map_data.moon_temp[mask] * beam) * resolution**2
            # Moon center effective brightness temperature
            #cent_bt  = moon_beam /  min(moon_solid_angle, beam_solid_angle)
            # Moon center brightness temperature
            cent_bt = moon_beam / (np.sum(beam) * resolution**2)
            # Moon center beam convolved  antenna temperature
            cent_at = moon_beam / beam_solid_angle
            return cent_at, cent_bt
    
    def moon_sim_map(self, obs_bs=0, convolve=False, cmb=False, illum=False, illum_map=False, pol=False):
        '''
        Given the datetime object, simulate the Moon temperature map with data from Chang'E satellite scaled
        down in mean temperature to agree with the Krotikov & Pelyushenko (1987) model.
        Other bands use same data scaled up by the ratio of the amplitude of the temperature variation in
        the Krotikov & Pelyushenko (1987) model to their 37 GHz amplitude and also scaled to the mean 
        temperature at zero latitude in the same model with the accompanying reduced phase offset.
        The default observatory is the CLASS site
        obs_bs provides the telescope boresight (in deg) during the observation
        map_size specifies the size of the* moon_ square map (in deg)
        res specifies the resolution of the map (in deg), can divide the map_size
        fwhm is the size of the beam
        convolve controls whether the map is convolved with the beam
        illum_map returns cartoon illuminatin amp if True
        pol returns polarization map if True
        Return a 2-D (N, N) arrays with as the maps (None for non-chosen options)
        '''
        # Detailed Moon Temp Map from Chang'E Satellite Data
        b0 = np.array( [270.87558, 0.61600, -7.563E-03, -4.700E-05, 3.013E-07, 1.275E-09, -3.988E-12, -1.275E-14])
        b20 = np.array([265.34897, 0.56421, -7.875E-03, -3.613E-05, 3.350E-07, 6.763E-10, -4.625E-12, -3.075E-15])
        b40 = np.array([247.42729, 0.52149, -6.838E-03, -3.988E-05, 2.900E-07, 1.113E-09, -3.975E-12, -1.205E-14])
        b60 = np.array([216.04583, 0.45040, -5.200E-03, -3.538E-05, 2.263E-07, 9.950E-10, -3.188E-12, -1.045E-14])
        #Scaled data for 87 GHz
        c0 = np.array( [268.51860, 0.62504, -0.00767, -4.77E-05, 3.06E-07, 1.29E-09, -4.05E-12, -1.29E-14])
        c20 = np.array([262.99199, 0.57249, -0.00799, -3.67E-05, 3.40E-07, 6.86E-10, -4.69E-12, -3.12E-15])
        c40 = np.array([245.07031, 0.52914, -0.00694, -4.05E-05, 2.94E-07, 1.13E-09, -4.03E-12, -1.22E-14])
        c60 = np.array([213.68885, 0.45701, -0.00528, -3.59E-05, 2.30E-07, 1.01E-09, -3.23E-12, -1.06E-14])
        #Scaled data for 150 GHz
        d0 = np.array( [274.21189, 0.76442, -0.00938, -5.83E-05, 3.74E-07, 1.58E-09, -4.95E-12, -1.58E-14])
        d20 = np.array([268.68528, 0.70016, -0.00977, -4.48E-05, 4.16E-07, 8.39E-10, -5.74E-12, -3.82E-15])
        d40 = np.array([250.76360, 0.64714, -0.00848, -4.95E-05, 3.60E-07, 1.38E-09, -4.93E-12, -1.50E-14])
        d60 = np.array([219.38214, 0.55892, -0.00645, -4.39E-05, 2.81E-07, 1.23E-09, -3.96E-12, -1.30E-14])
        #Scaled data for 220 GHz
        e0 = np.array( [278.70844, 0.88368, -0.01085, -6.74E-05, 4.32E-07, 1.83E-09, -5.72E-12, -1.83E-14])
        e20 = np.array([273.18183, 0.80939, -0.01130, -5.18E-05, 4.81E-07, 9.70E-10, -6.63E-12, -4.41E-15])
        e40 = np.array([255.26015, 0.74810, -0.00981, -5.72E-05, 4.16E-07, 1.60E-09, -5.70E-12, -1.73E-14])
        e60 = np.array([223.87869, 0.64612, -0.00746, -5.07E-05, 3.25E-07, 1.43E-09, -4.57E-12, -1.50E-14])

        def lat_temp(ha, b):
            #ha is the hour angle of the Sun on the Moon in degrees
            # -90 is sunrise, +90 is sunset
            #b contains the coefficients at a certain latitude
            ha = np.array([ha,])
            ha[ha > 180] -= 360
            ha[ha < -180] += 360
            tb = np.zeros(len(ha)) #brightness temperature
            #This will avoid a lot of round off error
            tb = b[7]
            for i in range(6, -1, -1):
                tb = ha * tb + b[i]
            return tb[0]
        def calc_ha_lat(x, y, radius, offset):
            #output hour angle and latitude (absolute value) given
            #x, y on a map.
            #Offset is the origin of hour angle
            #Note: Mirroring happened in coordinate system conversion.
            R = radius #Moon radius in deg
            lat = np.arcsin(y/R)
            R_prime = R*np.cos(lat)
            ha = np.degrees(np.arcsin(x/R_prime)) - offset
            return ha, abs(np.degrees(lat))
        def calc_tb(recv, x, y, radius, offset, rot):
            #calculate the brightness temperature of given x, y
            #offset gives the offset hour angle (sub-solar longitude of the Moon) in degrees
            #rot gives the rotation of the Moon on the sky (right hand rule of the map) in degrees
            rot_rad = np.radians(rot)
            sin_rot = np.sin(rot_rad)
            cos_rot = np.cos(rot_rad)
            x_p =  x * cos_rot + y * sin_rot
            y_p = -x * sin_rot + y * cos_rot
            ha, lat = calc_ha_lat(x_p, y_p, radius, offset)
            if recv == 'q_band':
                if lat < 10:
                    tb = lat_temp(ha, b0)
                elif lat < 30:
                    tb = lat_temp(ha, b20)
                elif lat < 50:
                    tb = lat_temp(ha, b40)
                else:
                    tb = lat_temp(ha, b60)
            elif recv == 'w1_band':
                ha_p = ha + 12.0
                if ha_p > 182: ha_p -= 349.0
                if lat < 10:
                    tb = lat_temp(ha_p, c0)
                elif lat < 30:
                    tb = lat_temp(ha_p, c20)
                elif lat < 50:
                    tb = lat_temp(ha_p, c40)
                else:
                    tb = lat_temp(ha_p, c60)
            elif recv == 'gl_band':
                ha_p = ha + 18.0
                if ha_p > 185: ha_p -= 347.0
                if lat < 10:
                    tb = lat_temp(ha_p, d0)
                elif lat < 30:
                    tb = lat_temp(ha_p, d20)
                elif lat < 50:
                    tb = lat_temp(ha_p, d40)
                else:
                    tb = lat_temp(ha_p, d60)
            elif recv == 'gh_band':
                ha_p = ha + 22.0
                if ha_p > 186: ha_p -= 344.0
                if lat < 10:
                    tb = lat_temp(ha_p, e0)
                elif lat < 30:
                    tb = lat_temp(ha_p, e20)
                elif lat < 50:
                    tb = lat_temp(ha_p, e40)
                else:
                    tb = lat_temp(ha_p, e60)
            else:
                raise ValueError('Invalid receiver.')    

            return tb
        def calc_illum(x, y, radius, offset, rot):
            #calculate cartoon of illumination given x, y
            #offset gives the offset hour angle (phase angle of the Moon) in degrees
            #rot gives the rotation of the Moon on the sky (right hand rule of the map) in degrees
            rot_rad = np.radians(rot)
            sin_rot = np.sin(rot_rad)
            cos_rot = np.cos(rot_rad)
            x_p =  x * cos_rot + y * sin_rot
            y_p = -x * sin_rot + y * cos_rot
            ha, lat = calc_ha_lat(x_p, y_p, radius, offset)
            ha = np.array([ha,])
            ha[ha > 180] -= 360
            ha[ha < -180] += 360
            #illumination (dark = 30, light = 100)
            if abs(ha) < 90:
                illum = 100
            else:
                illum = 30
            return illum
        def pol_frac(r, radius, eps=1.9):
            '''Calculate Moon polarization fraction at certain radius
            r is the radius to query (in deg)
            radius is the radius of the Moon (in deg)
            eps is the dielectric constant, taking 1.9 for now
            Returns the polarization fraction
            '''
            alpha = np.arcsin(r/radius)
            a = np.cos(alpha)
            b = np.sqrt(eps - np.sin(alpha)**2)
            r_s = ((a - b)/(a + b))**2
            a = np.sqrt(eps)*np.cos(alpha)
            b = np.sqrt(1 - np.sin(alpha)**2/eps)
            r_p = ((a - b)/(a + b))**2
            t_s = 1 - r_s
            t_p = 1 - r_p
            return 0.5*(t_p - t_s)    

        N = int(self.map_size/self.res)
        temp_bg = 2.725 #K of the CMB
        
        #Moon Phase/Radius Calculation
        if illum_map: illum = True
        phase, sslong, rot_p, rot_bl = self.moon_phase_rot(obs_bs, illum)
        #Scale map according to the Moon's radius
        scale = self.map_data.radius / self.map_data.ref_radius
        self.map_data.res = self.res * scale
        self.map_data.map_size = self.map_size * scale

        x = np.linspace(-self.map_data.map_size/2., self.map_data.map_size/2., N, endpoint=False) + self.map_data.res/2.
        y = np.linspace(-self.map_data.map_size/2., self.map_data.map_size/2., N, endpoint=False) + self.map_data.res/2.
        self.map_data.x, self.map_data.y = np.meshgrid(x, y, indexing='ij')
        r = np.sqrt(self.map_data.x**2 + self.map_data.y**2)

        idx = np.array(np.where(r <= self.map_data.radius))
        if cmb:
            self.map_data.moon_temp = np.full((N, N), temp_bg)
        else:
            self.map_data.moon_temp = np.full((N, N), 0.)
        if pol:
            self.map_data.moon_u = np.zeros((N, N))
        if illum_map:
            self.map_data.moon_illum = np.zeros((N, N))

        for i0, i1 in idx.T:
            x_t = self.map_data.x[i0, i1]
            y_t = self.map_data.y[i0, i1]
            r_t = r[i0, i1]
            self.map_data.moon_temp[i0, i1] = calc_tb(self.recv, x_t, y_t, self.map_data.radius, sslong, rot_p) #Unit: Kelvin
            if illum_map:
                self.map_data.moon_illum[i0, i1] = calc_illum(x_t, y_t, self.map_data.radius, phase, rot_bl)
            if pol:
                chi = np.arctan2(y_t, x_t)
                self.map_data.moon_u[i0, i1] = self.map_data.moon_temp[i0, i1] * pol_frac(r_t, self.map_data.radius) * np.sin(2*chi)
        if convolve:
            #beam = gauss_map_gen(map_size=map_size, res=res)
            sigma = self.fwhm/(self.map_data.res * SIG2FWHM)
            self.map_data.moon_temp = gaussian_filter(self.map_data.moon_temp, sigma)
            if pol:
                self.map_data.moon_u = gaussian_filter(self.map_data.moon_u, sigma)
        return self.map_data

    def moon_phase_rot(self, obs_bs, illum=False):
        '''Calculate the Moon phase angle, sub-solar longitude and rotation in the receiver coordinates
        '''
        self.moon_phase_pa(illum)
        #No problem with the NLP, just subtract the parallactic and boresight angles
        rot_p =  np.degrees(self.map_data.pa_p - self.map_data.par_ang) - obs_bs #deg
        if illum:
            #Rotate bright limb vector 90 degrees to align with its pole.
            #Note that this is NOT aligned with the NLP.
            rot_bl = np.degrees(self.map_data.pa_bl - self.map_data.par_ang) - obs_bs + 90.0 #deg  
            #convert phase angle to degrees
            phase = np.degrees(self.map_data.phase)
        else:
            #dummy return value
            phase = None
            rot_bl = None

        return phase, np.degrees(self.map_data.sslong), rot_p, rot_bl

    def moon_pole_angle(self, geocentric=False):
        '''
        Returns topocentric or geocentric position angle of the lunar pole wrt the NCP (positive CCW on the sky)
        Ignoring nutations and physical librations. Should be good to about +/-2 arcminutes. 
        Ref: Computation of the Quantities Describing the Lunar Librations in The Astronomical Almanac, D.B. Taylor et al.
             Explanatory Supplement to the Astronomical Almanac, P.K. Seidelmann
        '''
        if self.dt is None:
            raise ValueError('A time must be specified.')
        sec_to_rad = np.pi / 648000.0
        dtai = 37.0
        if   self.dt < datetime(2012, 7, 1): dtai = 34.0
        elif self.dt < datetime(2015, 7, 1): dtai = 35.0
        elif self.dt < datetime(2017, 7, 1): dtai = 36.0
        deltaT = 32.184 + dtai
        jd_tt = ephem.julian_date(self.dt + timedelta(seconds=deltaT))
        t = jd_tt - ephem.julian_date(ephem.J2000)
        #Centuries since J2000
        t /= 36525.0
        #longitude of moon's ascending node
        omega = 450160.280 + (-482890.539 + (7.455 + 0.008 * t) * t) * t - 1296000.0 * ((5.0 * t) % 1.0) 
        omega = (omega * sec_to_rad) % (2.0 * np.pi)
        #mean obliquity of ecliptic   
        eps = 84381.448 + (-46.8150 + (-0.00059 + 0.001813 * t) * t) * t
        eps = eps * sec_to_rad
        #inclination of the ecliptic to the mean lunar equator
        I = 1.542666667 * np.pi / 180.0
        sin_I = np.sin(I)
        cos_I = np.cos(I)
        sin_eps = np.sin(eps)
        cos_eps = np.cos(eps)
        sin_omega = np.sin(omega)
        cos_omega = np.cos(omega)
        #inclination of the mean equator of the Moon to the true equator of the Earth
        i = np.arccos(cos_I * cos_eps + sin_I * sin_eps * cos_omega)
        #distance from the true equinox of date to the ascending node of the mean equator 
        #of the Moon on the true equator of the Earth.  
        omega_prime = np.arctan2(-sin_I * sin_omega, cos_I * sin_eps - sin_I * cos_eps * cos_omega)
        sin_i = np.sin(i)
        if geocentric:
            dec = self.moon.g_dec
            ra = self.moon.g_ra
        else:
            dec = self.moon.dec
            ra = self.moon.ra
        sin_m_dec = np.sin(dec)
        cos_m_dec = np.cos(dec)
        #position angle of the lunar pole CCW wrt NCP
        c_prime_o = np.arctan2(-sin_i * np.cos(omega_prime - ra), 
                              cos_m_dec * np.cos(i) - sin_m_dec * sin_i * np.sin(omega_prime - ra))
        return c_prime_o

    def moon_phase_pa(self, illum=True, geocentric=False):
        '''
        Computes the Moon phase, sub-solar longitude, NLP position angle and parallactic angle
        Bright limb position angle option
        Geocentric option (mainly for checking calculations) 
        return phase, sub-solar longitude and position angles (in radians), Moon radius in degrees
        '''
        if self.dt is None:
            raise ValueError('A time must be specified.')
        if geocentric:
            m_dec = self.moon.g_dec
            m_ra  = self.moon.g_ra
            s_dec = self.sun.g_dec
            s_ra  = self.sun.g_ra
        else:
            m_dec = self.moon.dec
            m_ra  = self.moon.ra
            s_dec = self.sun.dec
            s_ra  = self.sun.ra
        sin_m_dec = np.sin(m_dec)
        cos_m_dec = np.cos(m_dec)
        sin_s_dec = np.sin(s_dec)
        cos_s_dec = np.cos(s_dec)
        cos_dra = np.cos(s_ra - m_ra)
        sin_dra = np.sin(s_ra - m_ra)
        #Moon's elongation from the Sun.
        elong = np.arccos(sin_s_dec * sin_m_dec + cos_s_dec * cos_m_dec * cos_dra)
        #Moon sub-solar longitude used for brightness temperature 
        # (co-longitude is the west longitude of the morning terminator)
        self.map_data.sslong = np.pi / 2.0 - self.moon.colong
        #Keep it between +/-180 (colong ranges from 0 to 360)
        if self.map_data.sslong < -np.pi: self.map_data.sslong += 2.0 * np.pi
        #Position angle of the NLP
        self.map_data.pa_p = self.moon_pole_angle(geocentric)
        if illum:
            #Moon phase angle calculation used for illumination.
            self.map_data.phase = np.arctan2(self.sun.earth_distance * np.sin(elong),
                                  self.moon.earth_distance - self.sun.earth_distance * np.cos(elong))
            #Position angle of the bright limb wrt NCP  (CCW positive on the sky)
            #Positive cooresponds to a waning moon, negative to a waxing moon
            self.map_data.pa_bl  = np.arctan2(cos_s_dec * sin_dra, 
                                   sin_s_dec * cos_m_dec - cos_s_dec * sin_m_dec * cos_dra)
            self.map_data.illum_frac = 0.5 * (1.0 + np.cos(self.map_data.phase))
            #Negative for waning moon
            if self.map_data.pa_bl > 0.0: self.map_data.illum_frac *= -1.0
            #print ('elong', np.degrees(elong), 'ephem_elong', np.degrees(moon.elong), 'libration_long', np.degrees(moon.libration_long), 
            #       'sslong', np.degrees(sslong),'phase', np.degrees(phase), 'pa_bl', np.degrees(pa_bl))
        lat = self.ephem.site.lat
        cos_lat = np.cos(lat)
        lst = self.ephem.sidereal_time()
        ha = lst - self.moon.ra
        if geocentric:
        # parallactic angle is never geocentric, of course
            cos_m_dec = np.cos(self.moon.dec)
            sin_m_dec = np.sin(self.moon.dec)
        self.map_data.par_ang = np.arctan2(cos_lat * np.sin(ha), 
                                np.sin(lat) * cos_m_dec - cos_lat * sin_m_dec * np.cos(ha))
        #print 'hour angle', ephem.hours(ha), 'parallactic angle', np.degrees(par_ang)
        return self.map_data

