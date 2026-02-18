import numpy as np
import matplotlib.pyplot as plt
from collections import deque

class Muscle:
    def __init__(self, name, F_max_iso, l_opt, l_slack, v_max, N, K, l_ref, dt, w = 0.56, eps_ref = 0.04, max_force_history=100):
        self.name = name
        self.F_max_iso = F_max_iso  
        self.l_opt = l_opt        
        self.l_slack = l_slack     
        self.v_max = v_max          
        self.N = N                 
        self.K = K 
        self.w = w
        self.eps_ref = eps_ref
        self.l_ref = l_ref
        self.dt = dt
        self.l_ce = l_opt
        self.tau = 0.01 #update this based on the paper
        self.act = 0.0
        self.stim0 = 0.01 #baseline stimulation, update this based on the
        # Initialize deque for storing muscle forces (for delayed feedback)
        self.max_force_history = max_force_history
        self.force_history = deque(maxlen=max_force_history)

    def force_length_relationship_CE(self, l_ce):
        w = self.w
        l_opt = self.l_opt
        c = np.log(0.05) 

        f_l = np.exp(c*np.abs((l_ce - l_opt) / (l_opt*w))**3)     
        return f_l

    def force_velocity_relationship_CE(self, v_ce):
        v_max = self.v_max
        K = self.K
        N = self.N

        if v_ce <= 0:
            f_v = (v_max + v_ce) / (v_max - K * v_ce)   

        else:
            f_v = N  + (N-1)*(( - v_max + v_ce)/(7.56*K*v_ce + v_max))

        return f_v
    
    def force_length_relationship_SEE(self, l_see):
        l_slack = self.l_slack
        eps = (l_see - l_slack)/l_slack

        eps_ref = self.eps_ref 

        if eps > 0:
            f_see = (eps/eps_ref)**2
        else:
            f_see = 0

        return f_see
    
    def force_length_relationship_PE(self, l_ce):
        l_opt = self.l_opt

        if l_ce > l_opt:
            f_pe = ((l_ce/l_opt - 1) / self.w)**2
        else:
            f_pe = 0

        return f_pe

    def inverse_force_velocity_CE(self, f_v):
        v_max = self.v_max
        K = self.K
        N = self.N

        if f_v <= 1:
            v_ce = v_max * (f_v - 1) / (1 + K * f_v)    
        elif (f_v > 1)   and (f_v <= N):
            v_ce = (((f_v - 1) * v_max) / (N - 1- 7.56*K*(f_v - N)))
        else:
            v_ce = v_max*(1 + 0.01*(f_v - N))
        return v_ce
    
    def update_activation(self, stim):
        self.act = self.act + (stim - self.act) / self.tau * self.dt
        # Clamp activation to physiological range [0, 1]
        #self.act = np.clip(self.act, 0.0, 1.0)
        return self.act
    
    def get_muscle_dynamics(self, l_ce, l_m, stim):        
        """
        Computes the muscle force based on current muscle state and stimulation.
        This function would be called during the simulation loop to update muscle force.

        f_m = f_see_norm * self.F_max_iso
        f_ce = f_see_norm - f_pe_norm

        we calculate v_ce here and return f_see and v_ce. f_see is the muscle force (or basically f_m).

        """
        act = self.update_activation(stim)
        l_see = l_m - l_ce
        f_see_norm = self.force_length_relationship_SEE(l_see)
        f_pe_norm = self.force_length_relationship_PE(l_ce)

        f_l_ce_norm = self.force_length_relationship_CE(l_ce)

        f_ce = (f_see_norm - f_pe_norm) * self.F_max_iso

        if f_ce < 0:
            f_ce = 0

        denom = act * f_l_ce_norm * self.F_max_iso
        if denom < 1e-6:
            f_v = 1  # Update this later after discussing how to handle near-zero activation or force-length conditions (adding a small number may not be accurate)
        else:
            f_v = f_ce / denom #to avoid division by zero
        v_ce = self.inverse_force_velocity_CE(f_v)
        f_m = f_see_norm * self.F_max_iso
        # try clipping muscle force?
        #f_m = np.clip(f_m, 0, self.F_max_iso)
        return f_m, v_ce 
    
    def update_muscle_state(self, l_m, stim):
        """
        This function updates the muscle state (fiber length) based on current muscle length, activation, 
        and the force-velocity relationship. 
        It should be called in each time step of the simulation loop.

        """
        f_m, v_ce = self.get_muscle_dynamics(self.l_ce, l_m, stim)
        
        #l_ce_new = l_ce + v_ce * dt
        #we have to initialise self.l_ce somewhere before this function is called. 
        self.l_ce += v_ce * self.dt
        
        # 3. Store for plotting/feedback
        self.current_force = f_m
        self.current_velocity = v_ce
        
        # 4. Store force in deque for delayed feedback
        self.force_history.append(f_m)
        
        return f_m
    
    def get_delayed_force(self, delay_steps=1):
        """
        Get the muscle force from 'delay_steps' iterations ago.
        
        Parameters:
        -----------
        delay_steps : int
            Number of timesteps to look back in history (default: 1).
            delay_steps=1 returns the force from 1 timestep ago.
            
        Returns:
        --------
        float
            The delayed muscle force, or the oldest available if not enough history.
        """
        if delay_steps <= 0:
            return 0.0
        
        if delay_steps > len(self.force_history):
            # Return oldest available force if delay exceeds history length
            return self.force_history[0] if len(self.force_history) > 0 else 0.0
        
        # Convert deque to list and index from the end
        return self.force_history[-delay_steps]
    
    def get_force_history_array(self):
        """
        Get all stored forces as a numpy array.
        
        Returns:
        --------
        np.ndarray
            Array of stored muscle forces.
        """
        return np.array(list(self.force_history))
    
    def clear_force_history(self):
        """Clear the force history deque."""
        self.force_history.clear()
        
    def plot_characteristics(self):
        """
        Plots the muscle specific Force-Length and Force-Velocity curves based on 
        the equations defined in this class.
        """
        fig, ax = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(f"Muscle Characteristics: {self.name} (F_max={self.F_max_iso}N)")

        # 1. Force-Length (CE & PE)
        # Sweep fiber length from 50% to 180% of optimal length
        l_scan = np.linspace(0.5 * self.l_opt, 1.8 * self.l_opt, 100)
        
        # Calculate forces using YOUR methods
        fl_ce = [self.force_length_relationship_CE(l) * self.F_max_iso for l in l_scan]
        fl_pe = [self.force_length_relationship_PE(l) * self.F_max_iso for l in l_scan]
        fl_total = np.array(fl_ce) + np.array(fl_pe)

        ax[0].plot(l_scan, fl_ce, 'b--', label='Active (CE)')
        ax[0].plot(l_scan, fl_pe, 'g--', label='Passive (PE)')
        ax[0].plot(l_scan, fl_total, 'k-', lw=2, label='Total Isometric')
        ax[0].axvline(self.l_opt, color='r', alpha=0.3, label='L_opt')
        ax[0].set_title("Force-Length Relationship")
        ax[0].set_xlabel("Fiber Length (m)")
        ax[0].set_ylabel("Force (N)")
        ax[0].legend()
        ax[0].grid(True)

        # 2. Force-Length (Tendon / SEE)
        # Sweep tendon length slightly above slack length
        l_see_scan = np.linspace(self.l_slack, 1.08 * self.l_slack, 100)
        fl_see = [self.force_length_relationship_SEE(l) * self.F_max_iso for l in l_see_scan]

        ax[1].plot(l_see_scan, fl_see, 'm-', lw=2)
        ax[1].axvline(self.l_slack, color='r', alpha=0.3, label='L_slack')
        ax[1].set_title("Tendon Stiffness (SEE)")
        ax[1].set_xlabel("Tendon Length (m)")
        ax[1].grid(True)

        # 3. Force-Velocity
        # Sweep velocity from Shortening (-v_max) to Lengthening (+0.5*v_max)
        # Note: Your math implies negative v_ce is shortening (f_v < 1)
        v_scan = np.linspace(-self.v_max, 0.5 * self.v_max, 100)
        fv_curve = [self.force_velocity_relationship_CE(v) * self.F_max_iso for v in v_scan]

        ax[2].plot(v_scan, fv_curve, 'r-', lw=2)
        ax[2].axhline(self.F_max_iso, color='k', ls='--', alpha=0.5, label='F_max')
        ax[2].axvline(0, color='k', lw=1)
        ax[2].set_title("Force-Velocity Relationship")
        ax[2].set_xlabel("Velocity (m/s) [-Shortening / +Lengthening]")
        ax[2].set_ylabel("Force (N)")
        ax[2].grid(True)
        
        plt.tight_layout()
        plt.show()


    