import numpy as np
import unyt as u
import gvar as gv
import matplotlib.pyplot as plt
import numpy.testing as npt

#my_planet = (M, a, eps, phi, omega)
G = u.physical_constants.G.value    # Gravitational constant
M_s = u.physical_constants.msun.value    # Mass of Sun
mercury = (0.330104e24, 0.38709843 * 1.496e11, 0.20563661, 19.08833658 * np.pi/180, 77.45771895 * np.pi/180)
venus = (4.86732e24, 0.72332102 * 1.496e11, 0.00676399, 286.68776598 * np.pi/180, 131.76755713 * np.pi/180)
earth = (5.97219e24, 1.00000018 * 1.496e11, 0.01673163, 331.16416467 * np.pi/180, 102.93005885 * np.pi/180)
mars = (0.641693e24, 1.52371243 * 1.496e11, 0.09336511, 326.20022408 * np.pi/180, 23.91744784 * np.pi/180)
jupiter = (1898.13e24, 5.20248019 * 1.496e11, 0.04853590, 242.48991503 * np.pi/180, 14.27495244 * np.pi/180)
saturn = (568.319e24, 9.54149883 * 1.496e11, 0.05550825, 281.55831164 * np.pi/180, 92.86136063 * np.pi/180)
uranus = (86.8103e24, 19.18797948 * 1.496e11, 0.04685740, 29.32733579 * np.pi/180, 172.43404441 * np.pi/180)
neptune = (102.41e24, 30.06952752 * 1.496e11, 0.00895439, 1.14513741 * np.pi/180, 46.68158724 * np.pi/180)
planets = (mercury, venus, earth, mars, jupiter, saturn, uranus, neptune)

pluto = (0, 39.48211675 * 1.496e11, 0.24882730, 265.9093415 * np.pi/180, 224.06891629 * np.pi/180)
halley = (0, 17.834 * 1.496e11, 0.96714, 0, 111.33 * np.pi/180)


def get_planet_r(planet):
    """
    Determines distance 'r' from the sun in meters.
    
    Arguments:
    =====
    * planet: Tuple containing information in the form 
        (M (in kg), a (in meters), eps, phi, omega)
        
    Returns:
    =====
    The distance from the center of the sun to the center of the planet in meters.
    
    Example usage:
    =====
    >> neptune = (102.41e24, 30.06952752 * 1.496e11, 0.00895439, 1.14513741 * np.pi/180, 46.68158724 * np.pi/180)
    >> neptune_distance = get_planet_r(neptune)
    
    """
    
    a = planet[1]    # Semimajor axis of orbit in meters
    eps = planet[2]    # Eccintricity of orbit
    phi = planet[3]    # Orbital phase
    omega = planet[4]    # Determines the location of perihelion
    r = (a*(1-eps**2))/(1 + eps*np.cos(phi - omega))    # Distance from the sun in meters
    
    return r


def get_planet_coords(planet):
    """
    Determines a planets coordinates in the solar system (in 3-D cartesian)
    
    Arguments:
    =====
    * planet: Tuple containing information in the form 
        (M (in kg), a (in meters), eps, phi, omega)
        
    Returns:
    =====
    Coordinates for a planet in 3-D cartesian in meters
    
    Example usage:
    =====
    >> jupiter = (1898.13e24, 5.20248019 * 1.496e11, 0.04853590, 242.48991503 * np.pi/180, 14.27495244 * np.pi/180)
    >> jupiter_coords = get_planet_coords(jupiter)
    
    """
    
    r = get_planet_r(planet)    # Distance from sun in meters
    phi = planet[3]    # Orbital phase
    x = r * np.cos(phi)    # x-position in meters
    y = r * np.sin(phi)    # y-position in meters
    z = 0    # z-position in meters
    position = (x,y,z)
    
    return position


def get_planet_orbit(planet, phi_linspace):
    """
    Place holder (returns tuple of two arrays for x and y)
    Determines orbit from planet information and NumPy array of phi values.
    
    Arguments:
    =====
    * planet: Tuple containing information in the form 
        (M (in kg), a (in meters), eps, phi, omega)
    * phi_linspace: NumPy array of phi values
    
    Returns:
    =====
    A tuple of the form (x_array, y_array)
    
    "x_array" contains the x coordinates for the given phi value in meters as a NumPy array
    "y_array" contains the y coordinates for the given phi value in meters as a NumPy array
    
    Example usage:
    =====
    
    >> phi_linspace = np.linspace(0,2*np.pi,1000)
    >> mars = (0.641693e24, 1.52371243 * 1.496e11, 0.09336511, 326.20022408 * np.pi/180, 23.91744784 * np.pi/180)
    >> x_array, y_array = get_planet_orbit(mars, phi_linspace)
    
    """
    
    x_coords = ()
    y_coords = ()
    for i in phi_linspace:
        M = planet[0]    # Mass of planet in Kg
        a = planet[1]    # Semimajor axis of orbit in meters
        eps = planet[2]    # Eccintricity of orbit
        omega = planet[4]    # Determines the location of perihelion
        planet_new = (M, a, eps, i, omega)    
        w = get_planet_coords(planet_new)
        x_coords += w[0],
        y_coords += w[1],
        
    return x_coords, y_coords


def update_planet_position(planet, dt):
    """
    Takes a planet tuple and updates planet's position from change in time
    
    Arguments:
    =====
    * planet: Tuple containing information in the form 
        (M (in kg), a (in meters), eps, phi, omega)
    * dt: Change in time in seconds
    
    Returns:
    =====
    The original planet tuple with a new phi value
    
    Example usage:
    =====
    >> earth = (5.97219e24, 1.00000018 * 1.496e11, 0.01673163, 331.16416467 * np.pi/180, 102.93005885 * np.pi/180)
    >> dt = 100
    >> earth_new_position = update_planet_position(earth, dt)
    
    """
    
    M = planet[0]    # Mass of planet in Kg
    a = planet[1]    # Semimajor axis of orbit in meters
    eps = planet[2]    # Eccintricity of orbit
    phi = planet[3]    # Orbital phase
    omega = planet[4]    # Determines the location of perihelion
    r = get_planet_r(planet)
    dphi = dt * (np.sqrt(G * M_s * a * (1 - eps**2)))/(r**2)
    planet_new = (M, a, eps, phi + dphi, omega)
    
    return planet_new


def update_all_planets(planets, dt):
    """
    Updates position for all planets in tuple for given change in time.
    
    Arguments:
    =====
    * planets: Tuple containing planet tuples of the form
        (M (in kg), a (in meters), eps, phi, omega)
    * dt: Change in time in seconds
    
    Returns:
    =====
    The original tuple with the phi (position) value in each planet tuple update.
    
    Example usage:
    =====
    >> mercury = (0.330104e24, 0.38709843 * 1.496e11, 0.20563661, 19.08833658 * np.pi/180, 77.45771895 * np.pi/180)
    >> venus = (4.86732e24, 0.72332102 * 1.496e11, 0.00676399, 286.68776598 * np.pi/180, 131.76755713 * np.pi/180)
    >> planets = (mercury, venus)
    >> dt = 900
    >> planets_update = update_all_planets(planets, dt)
    
    """
    
    planet_updates = []
    for i in planets:
        planet_updates += [update_planet_position(i, dt)]
    return planet_updates


def accel_g_sun(vec_r):
    """
    Determines gravitational acceleration due to the Sun.
    
    Arguments:
    =====
    * vec_r: Positional vector of the form (x,y,z) in meters
    
    Returns:
    =====
    Acceleration due to the Sun at that point as a NumPy three-vector in m/s^2
    
    Example usage:
    =====
    >> vec_r = np.array([3,14,39])
    >> sun_accel = accel_g_sun(vec_r)
    
    """
    
    vec_mag = np.sqrt(vec_r[0]**2 + vec_r[1]**2 + vec_r[2]**2)    # Magnitude of vec_r
    phi = np.arctan(vec_r[1]/vec_r[0])    # Initial phi angle
    r = np.sqrt(vec_r[0]**2 + vec_r[1]**2)
    theta = np.arcsin(r/vec_mag)    # Initial theta angle
    accel_sun = -((G*M_s)/vec_mag**3)*vec_mag
    a_s_x = accel_sun * np.sin(theta) * np.cos(phi)
    a_s_y = accel_sun * np.sin(theta) * np.sin(phi)
    a_s_z = accel_sun * np.cos(theta)
    sun_a_vector = [a_s_x, a_s_y, a_s_z]
    sun_a_vector = np.array(sun_a_vector)

    return sun_a_vector


def accel_g_planet(vec_r, planet):
    """
    Determines gravitational acceleration due to a planet.
    
    Arguments:
    =====
    * vec_r: Positional vector of the form (x,y,z) in meters
    
    Returns:
    =====
    Acceleration due to a planet at the point vec_r as a NumPy three-vector in m/s^2
    
    Example usage:
    =====
    >> vec_r = np.array([3,14,39])
    >> saturn = (568.319e24, 9.54149883 * 1.496e11, 0.05550825, 281.55831164 * np.pi/180, 92.86136063 * np.pi/180)
    >> planet_accel = accel_g_planet(vec_r)
    
    """
    
    planet_vec = np.array(get_planet_coords(planet))
    M = planet[0]    # Mass of planet in kg
    distance = vec_r - planet_vec    # Difference between given vector and planet vector
    vec_mag = np.sqrt(distance[0]**2 + distance[1]**2 + distance[2]**2)
    phi = np.arctan(distance[1]/distance[0])
    r = np.sqrt(distance[0]**2 + distance[1]**2)
    theta = np.arcsin(r/vec_mag)
    
    distance = np.sqrt(distance[0]**2 + distance[1]**2 + distance[2]**2)
    accel_planet = -((G*M)/(abs(distance)**3)) * (distance)
    
    a_p_x = accel_planet * np.sin(theta) * np.cos(phi)
    a_p_y = accel_planet * np.sin(theta) * np.sin(phi)
    a_p_z = accel_planet * np.cos(theta)
    planet_a_vector = [a_p_x, a_p_y, a_p_z]
    planet_a_vector = np.array(planet_a_vector)
    
    return planet_a_vector


def update_position(vec_r, vec_v, planets, dt):
    """
    Updates position and velocity vectors using a tuple containing the planets of the solar system for a given time interval
    
    Arguments:
    =====
    * vec_r: Positional vector of the form (x,y,z) in meters
    * vec_v: Velocity vector of the form (x,y,z) in meters/second
    * planets: Tuple containing planet tuples of the form
        (M (in kg), a (in meters), eps, phi, omega)
    * dt: Change in time in seconds
    
    Returns:
    =====
    Update velocity and position vectors
    
    "vec_v_new" is the updated velocity vector
    "vec_r_new" is the updated position vector
    
    Example usage:
    =====
    >> mercury = (0.330104e24, 0.38709843 * 1.496e11, 0.20563661, 19.08833658 * np.pi/180, 77.45771895 * np.pi/180)
    >> venus = (4.86732e24, 0.72332102 * 1.496e11, 0.00676399, 286.68776598 * np.pi/180, 131.76755713 * np.pi/180)
    >> earth = (5.97219e24, 1.00000018 * 1.496e11, 0.01673163, 331.16416467 * np.pi/180, 102.93005885 * np.pi/180)
    >> mars = (0.641693e24, 1.52371243 * 1.496e11, 0.09336511, 326.20022408 * np.pi/180, 23.91744784 * np.pi/180)
    >> jupiter = (1898.13e24, 5.20248019 * 1.496e11, 0.04853590, 242.48991503 * np.pi/180, 14.27495244 * np.pi/180)
    >> saturn = (568.319e24, 9.54149883 * 1.496e11, 0.05550825, 281.55831164 * np.pi/180, 92.86136063 * np.pi/180)
    >> uranus = (86.8103e24, 19.18797948 * 1.496e11, 0.04685740, 29.32733579 * np.pi/180, 172.43404441 * np.pi/180)
    >> neptune = (102.41e24, 30.06952752 * 1.496e11, 0.00895439, 1.14513741 * np.pi/180, 46.68158724 * np.pi/180)
    >> planets = (mercury, venus, earth, mars, jupiter, saturn, uranus, neptune)
    vec_r = np.array([1 * 1.496e11, -5 * 1.496e11,0])
    vec_v = np.array([1,-20,50])
    dt = 1e4
    v_new, r_new = update_position(vec_r, vec_v, planets, dt)
    
    """
    
    sun_accel = accel_g_sun(vec_r)
    planets_accel = []
    for i in planets:
        planets_accel += accel_g_planet(vec_r, i),

    planets_accel = sum(planets_accel)
    dv = dt * (sun_accel + planets_accel)
    vec_v_new = vec_v + dv
    dr = vec_v_new * dt
    vec_r_new = vec_r + dr
    
    return vec_r_new, vec_v_new


def get_planet_distances(vec_r, planets):
    """
    Creates a NumPy array of the distances from the given position vector to each planet in planets in order.
    
    Arguments:
    =====
    * vec_r: Positional vector in 3D cartesian coordinates in meters
    * planets: Tuple containing planet tuples of the form
        (M (in kg), a (in meters), eps, phi, omega)
    
    Returns:
    =====
    An array of the distances from the given position vector to each planet in meters
    
    Example usage:
    =====
    >> vec_r = np.array([0,0,0])
    >> mercury = (0.330104e24, 0.38709843 * 1.496e11, 0.20563661, 19.08833658 * np.pi/180, 77.45771895 * np.pi/180)
    >> venus = (4.86732e24, 0.72332102 * 1.496e11, 0.00676399, 286.68776598 * np.pi/180, 131.76755713 * np.pi/180)
    >> planets = (mercury, venus)
    >> distances = get_planet_distances(vec_r, planets):
    
    """
    
    planet_positions = []
    for i in planets:
        planet_positions += [get_planet_coords(i)]
    planet_positions = np.array(planet_positions)
    
    vec_rp = []
    for i in planet_positions:
        vec_rp += [abs(vec_r - i)]
    vec_rp = np.array(vec_rp)
    f = []
    for i in vec_rp:
        f += [np.sqrt(i[0]**2 + i[1]**2 + i[2]**2)]
        
    return f


import numpy.testing as npt

def run_API_tests():
    """
    Runs various tests to make sure various functions are working as expected.
    
    Test 1:
    =====
    Checks to see that get_planet_r() and get_planet_distances() return same values from the origin.
    
    Test 2:
    =====
    Checks to see if Earth's gravity for accel_g_planet() matches known value.
    
    Test 3:
    =====
    Checks to see if accel_g_sun() and accel_g_planet() return the same value for a planet at the origin with
        the same mass as the Sun.
        
    Test 4:
    =====
    Tests update_planet_position() assure position only changes marginally for small dt, but does indeed change.
    
    Test 5:
    =====
    Makes sure accel_g_planets() can handle vectors of 0 for all planets.
    
    Test 6:
    =====
    
    
    Returns:
    =====
    None if all tests are passed, otherwise errors are raised.
    
    """
    
    ### Test 1
    test_r = np.array([0,0,0])
    planet_distances = get_planet_distances(test_r, planets)
    planet_rs = []
    for i in planets:
        planet_rs += get_planet_r(i),
    npt.assert_allclose(planet_distances, planet_rs, rtol=1e-3)
    
    ## Test 2
    test = get_planet_coords(earth)
    test = np.array(test)
    test[0] = test[0] + 1
    test[1] = test[1] + 6.371e6
    test[2] = test[2] + 1
    npt.assert_allclose(accel_g_planet(test, earth)[1], -9.8, rtol=1e-1)
    
    ## Test 3
    test_vec = np.array([4,2,1])
    test_planet = (u.physical_constants.msun.value, 0,0,0,0)
    npt.assert_allclose(accel_g_planet(test_vec, test_planet), accel_g_sun(test_vec), rtol=1e-10)
    
    ## Test 4
    dt = 1
    npt.assert_allclose(update_planet_position(earth, dt), earth, rtol=1e-5)
    assert update_planet_position(earth, dt) != earth
    
    ## Test 5
    vec_r = np.array([0,0,0])
    for i in planets:
        accel_g_planet(vec_r, i)
        
    ## Test 6
    test = get_planet_coords(earth)
    test = np.array(test)
    test[0] = test[0] + 6.371e6

    test2 = np.array(test)
    test2[0] = test[0] - 6.371e6

    npt.assert_allclose(accel_g_sun(test),accel_g_sun(test2),rtol=1e-3)
    
    return None
    
print(run_API_tests())

   
def find_trajectory(vec_r0, vec_v0, planets, t_steps):
    """
    Main loop for solar system gravitation project.
    
    Arguments:
    =====
    * vec_r0: Initial 3-vector position of the small mass (Cartesian coordinates.)  
    * vec_v0: Initial 3-vector velocity of the small mass (Cartesian coordinates.)
    * planets: a list of planet tuples, at their initial positions.
        A planet tuple has the form:
            (M, a, eps, phi, omega)
        where M is the planet's mass, phi is the planet's angular position, 
        and a, eps, omega are orbital parameters.
    * t_steps: NumPy array (linspace or arange) specifying the range of times to simulate
        the trajectory over, regularly spaced by timestep dt.
        
    Returns:
    =====
    A tuple of the form (traj, planet_distance).
    
    "traj" contains the coordinates (x,y,z) of the test mass at each 
    corresponding time in t_steps, as a (3) x (Nt) array.
    "planet_distance" contains the distances from the small mass
    to each planet in planets, in order, as a function of time - this is a
    (len(planets)) x (Nt) array.
    
    Example usage:  (using kg-AU-s units)
    =====
    >> import unyt as u
    >> earth = (5.97219e24, 1.0, 0.01673163, 5.779905, 1.88570)
    >> r0 = np.array([-0.224, 0.98, 0.0])  # AU
    >> v0 = np.array([2e-9, 0.0, 0.0]) # AU/s 
    >> t = (np.arange(0, 4*365) * u.day).to_value('s')  # evolve for 4 years
    >> traj, pd = find_trajectory(r0, v0, [earth], t)
    
    """
    
    dt = t_steps[1] - t_steps[0]
    Nt = len(t_steps)
    
    traj = np.zeros((3, Nt))
    traj[:,0] = vec_r0
    
    planet_distance = np.zeros((len(planets), Nt))
    planet_distance[:,0] = get_planet_distances(vec_r0, planets)
    
    vec_v = vec_v0
    
    for i in range(Nt-1):
        (traj[:,i+1], vec_v) = update_position(traj[:,i], vec_v, planets, dt)        
        planets = update_all_planets(planets, dt)
        planet_distance[:,i+1] = get_planet_distances(traj[:,i+1], planets)
        
    return (traj, planet_distance)


def velocity(item):
    """
    Determines the velocity for a small object from a given tuple
    
    Arguments:
    =====
    * item: a tuple of the form (M (in kg), a (in meters), eps, phi, omega) where M is presumed to be very small compared
        to the mass of the planets (ie. it's not used in the calculation)
    
    Returns:
    =====
    Velocity as a NumPy three vector in units of meters
    
    Example usage:
    =====
    >> pluto = (0, 39.48211675 * 1.496e11, 0.24882730, 265.9093415 * np.pi/180, 224.06891629 * np.pi/180)
    >> pluto_velocity = velocity(pluto)
    
    """

    a = item[1]    # Semimajor axis of orbit in meters
    eps = item[2]    # Eccintricity of orbit
    phi = item[3]    # Orbital phase
    omega = item[4]    # Determines the location of perihelion
    r = get_planet_r(pluto)
    velocity = ((a*(1-eps**2))/(1+eps*np.cos(phi-omega)))*((np.sqrt(G * M_s * a * (1 - eps**2)))/(r**2))*(-np.sin(phi)+np.cos(phi))
    v_x = velocity * np.cos(phi)
    v_y = velocity * np.sin(phi)
    v_z = 0
    vel_xyz = np.array([v_x, v_y, v_z])
    
    return vel_xyz


comets_data = open('comets.csv')

## Taken from a tutorial
def parse_line_csv(line):
    """
    Parses through .csv file to remove artifacts and convert everything to float values.
    """
    strip_line = line.strip()
    line_list = strip_line.split(',')
    number_list = []
    for x in line_list:
        number_list.append(float(x))
        
    return number_list

comets = []
for i in comets_data:
    comets += parse_line_csv(i),
comets = np.array(comets)    # Data from comets file properly converted

comet_vecs = []
comet_vels = []
for i in comets:
    comet_vecs += [i[1] * 1.496e11, i[2] * 1.496e11, i[3] * 1.496e11],
    comet_vels += [i[4]*1000, i[5]*1000, i[6]*1000],