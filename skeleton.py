import numpy as np

class Skeleton:
    def __init__(self, muscle):
        self.muscle = muscle
        self.mass = 80  # mass of the hopper in kg
        self.g = 9.81
        self.l_f = 0.99 #flight lenght of the leg, not sure this skeleton specifc or simulation specific.
        self.d = 0.04 #moment arm
        self.phi_ref = 110*np.pi/180 #reference knee angle in radians
        self.l_ref = 0.5 #update this 
        self.l_s = 0.5 #segment length of the leg, update this based on actual geometry of the leg.
        self.v = 0.0 #initial velocity of the hopper
        self.y = 1 #initial height of the hopper, start standing (leg almost fully extended)

    def get_leg_length(self, y, stance: bool):
        if stance:
            return y
        else:
            return self.l_f 

    def get_leg_force(self, f_m, l_leg):
        d = self.d #d is specidic to knee geometry, we can set it as a parameter in the Skeleton class.
        l_s = self.l_s
        denom = np.sqrt(l_s**2 - (l_leg / 2)**2)
        if denom < 1e-6:
            denom = 1e-6
        f_leg = (d / denom) * f_m
        return f_leg
    
    def get_muscle_length(self, phi):
        l_ref = self.l_ref
        phi_ref = self.phi_ref
        d = self.d

        l_m = l_ref - d*(phi - phi_ref)

        return l_m
    
    def get_joint_angle(self, l_leg):
        l_s = self.l_s

        arg = 1 - (l_leg**2) / (2 * l_s**2)
        #arg = np.clip(arg, -1.0, 1.0)
        phi = np.arccos(arg)

        return phi
    
    def update_state(self, y, v, f_leg, dt):
        m = self.mass
        g = self.g
        a = (f_leg/m) - g
        self.v = self.v + a*dt
        self.y = self.y + self.v*dt
    
        return self.y, self.v

    def update_muscle_length(self, f_m, l_leg, dt, stance):

        f_leg = self.get_leg_force(f_m, l_leg)
        y, v = self.update_state(self.y, self.v, f_leg, dt)
        l_leg = self.get_leg_length(y, stance)
        phi = self.get_joint_angle(l_leg)
        l_m = self.get_muscle_length(phi)
        
        return l_m



