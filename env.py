from muscle import Muscle
from skeleton import Skeleton
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Simulation:
    def __init__(self, muscle: Muscle, skeleton: Skeleton, dt=0.00005, t_max=3.0, animate=False, save_dir=None, save_video=False):
        self.muscle = muscle
        self.skeleton = skeleton
        self.dt = dt
        self.t_max = t_max
        self.time = np.arange(0, t_max, dt)
        self.y_history = []
        self.v_history = []
        self.f_leg_history = []
        self.G = 2.64/self.muscle.F_max_iso 
        self.animate = animate
        self.animation_results = None
        self.save_dir = save_dir
        self.save_video = save_video

    def get_stim(self, P, dt):
        stim = self.muscle.stim0  + self.G*P 
        stim = np.clip(stim, 0.0, 1.0) # ensure stimulation is between 0 and 1  
        return stim
    
    def get_phase(self) -> bool:
        """Check if hopper is in stance phase."""
        if self.skeleton.y <= self.skeleton.l_f:   
            return True
        else:
            return False
    
    def step(self, stim):
        """
        Advance the simulation by one timestep, updating internal muscle state.
        """
        stance = self.get_phase()
        
        # Determine leg length based on phase
        l_leg = self.skeleton.get_leg_length(self.skeleton.y, stance)
        
        # Get joint angle from leg length
        phi = self.skeleton.get_joint_angle(l_leg)
        
        # Calculate muscle length from joint angle
        l_m = self.skeleton.get_muscle_length(phi)
        
        # Update muscle state and get muscle force
        # Pass stim (not act) - activation will be updated inside update_muscle_state
        f_m = self.muscle.update_muscle_state(l_m, stim)
        
        # Calculate leg force from muscle force
        if stance:
            f_leg = self.skeleton.get_leg_force(f_m, l_leg)
        else:
            f_leg = 0.0  # No leg force during flight phase
        
        # Update hopper dynamics (updates self.skeleton.y and self.skeleton.v)
        self.skeleton.y, self.skeleton.v = self.skeleton.update_state(self.skeleton.y, self.skeleton.v, f_leg, self.dt)

        return f_leg, f_m            

    def run(self, delay_steps=0):
        """
        Run the complete hopping simulation.
        """
        # Initialize storage
        y_hist = []
        v_hist = []
        f_leg_hist = []
        f_m_hist = []
        stim_hist = []
        act_hist = []
        leg_length_hist = []
        phase_hist = []
        
        # Run simulation
        for t in self.time:
            # Check current phase BEFORE calculating feedback
            in_stance = self.get_phase()
            
            # Get delayed force for feedback (only used if in stance)
            if in_stance:
                # Use the most recent available force if delay buffer not yet full
                if delay_steps > 0 and len(self.muscle.force_history) >= delay_steps:
                    P = self.muscle.get_delayed_force(delay_steps)
                else:
                    # Use most recent force if delay buffer not full yet
                    P = 0.0  # No feedback if not enough history
        
                stim = self.get_stim(P, self.dt)
            else:
                # During flight
                stim = self.muscle.stim0
            
            # Take simulation step (updates internal state)
            f_leg, f_m = self.step(stim)
            
            # Store history
            y_hist.append(self.skeleton.y)
            v_hist.append(self.skeleton.v)
            f_leg_hist.append(f_leg)
            f_m_hist.append(f_m)
            stim_hist.append(stim)
            act_hist.append(self.muscle.act)
            phase_hist.append(self.get_phase())
            leg_length_hist.append(self.skeleton.get_leg_length(self.skeleton.y, self.get_phase()))
        
        results = {
            'time': self.time,
            'height': np.array(y_hist),
            'velocity': np.array(v_hist),
            'leg_force': np.array(f_leg_hist),
            'muscle_force': np.array(f_m_hist),
            'stimulation': np.array(stim_hist),
            'activation': np.array(act_hist),
            'leg_length': np.array(leg_length_hist),
            'phase': np.array(phase_hist)
        }
        
        # Create animation if True
        if self.animate:
            self.animation_results = results
            self.create_animation(results)
        return results
    
    def create_animation(self, results, skip_frames=200):
        """
        Create an animation of the hopping simulation with two-segment leg.
        """
        # Calculate knee positions for all frames
        l_s = self.skeleton.l_s
        knee_x_traj = []
        knee_y_traj = []
        foot_y_traj = []
        
        for i in range(len(results['height'])):
            y_hip = results['height'][i]
            stance = y_hip <= self.skeleton.l_f
            
            if stance:
                l_leg = y_hip
                foot_y = 0
                knee_y = l_leg / 2
                knee_x_term = l_s**2 - knee_y**2
                if knee_x_term < 0:
                    knee_x_term = 0
                knee_x = np.sqrt(knee_x_term)
            else:
                l_leg = self.skeleton.l_f
                foot_y = y_hip - l_leg
                knee_y = foot_y + l_leg / 2
                knee_x_term = l_s**2 - (l_leg/2)**2
                if knee_x_term < 0:
                    knee_x_term = 0
                knee_x = np.sqrt(knee_x_term)
            
            knee_x_traj.append(knee_x)
            knee_y_traj.append(knee_y)
            foot_y_traj.append(foot_y)
        
        # Create figure
        fig_anim = plt.figure(figsize=(10, 10))
        ax_anim = plt.subplot(111)
        
        ax_anim.set_xlim(-1.5, 1.5)
        ax_anim.set_ylim(-0.2, 1.5)
        ax_anim.set_aspect('equal')
        ax_anim.axhline(0, color='brown', lw=3, label='Ground')
        ax_anim.set_xlabel('X (m)')
        ax_anim.set_ylabel('Y (m)')
        ax_anim.set_title('Hopping Animation - Two Segment Leg')
        ax_anim.grid(True, alpha=0.3)
        
        # Shank segment (from foot to knee)
        shank, = ax_anim.plot([], [], 'b-', lw=6, label='Shank')
        # Thigh segment (from knee to hip)
        thigh, = ax_anim.plot([], [], 'r-', lw=6, label='Thigh')
        # Point mass at hip
        mass, = ax_anim.plot([], [], 'ko', ms=15, label='Mass (Hip)')
        
        time_text = ax_anim.text(0.02, 0.95, '', transform=ax_anim.transAxes, 
                                fontsize=12, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax_anim.legend(loc='upper right', fontsize=10)
        
        # for animation
        indices = np.arange(0, len(results['time']), skip_frames)
        
        def update_anim(frame):
            idx = indices[frame] if frame < len(indices) else indices[-1]
            if idx >= len(results['time']):
                idx = len(results['time']) - 1
            
            y_hip = results['height'][idx]
            foot_y = foot_y_traj[idx]
            knee_y = knee_y_traj[idx]
            knee_x = knee_x_traj[idx]
            
            shank.set_data([0, knee_x], [foot_y, knee_y])
    
            thigh.set_data([knee_x, 0], [knee_y, y_hip])
         
            mass.set_data([0], [y_hip])
            
            # Determine phase
            is_stance = y_hip <= self.skeleton.l_f
            phase_str = 'STANCE' if is_stance else 'FLIGHT'
            
            time_text.set_text(f'Time: {results["time"][idx]:.3f} s\n'
                             f'Height: {y_hip:.3f} m\n'
                             f'Velocity: {results["velocity"][idx]:.3f} m/s\n'
                             f'Phase: {phase_str}\n'
                             f'Force: {results["leg_force"][idx]:.1f} N')
            
            return thigh, shank, mass, time_text
        
        # Create animation
        n_frames = len(indices)
        ani = animation.FuncAnimation(fig_anim, update_anim, frames=n_frames,
                                     interval=10, blit=True, repeat=True)
        
        plt.tight_layout()
        
        if self.save_video:
            if not self.save_dir:
                raise ValueError("save_dir is required when save_video is True")
            video_path = os.path.join(self.save_dir, 'hopping_animation.mp4')
            print(f"Saving animation to {video_path}...")
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=60, metadata=dict(artist='Simulation'), bitrate=1800)
            ani.save(video_path, writer=writer)
            print(f"Animation saved successfully to {video_path}")
            plt.close(fig_anim)
        else:
            plt.show()
            plt.close(fig_anim)
        
        return ani


        

    
