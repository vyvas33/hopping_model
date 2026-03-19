from muscle import Muscle
from skeleton import Skeleton
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
        
        # Create animation if showing or saving video
        if self.animate or self.save_video:
            self.animation_results = results
            self.create_animation(results)
        return results
    
    def create_animation(self, results, skip_frames=200):
        """
        Create a polished animation of the hopping simulation with a full body.
        """
        # Precompute knee and foot positions for all frames
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
                knee_x = np.sqrt(max(knee_x_term, 0))
            else:
                l_leg = self.skeleton.l_f
                foot_y = y_hip - l_leg
                knee_y = foot_y + l_leg / 2
                knee_x_term = l_s**2 - (l_leg / 2)**2
                knee_x = np.sqrt(max(knee_x_term, 0))

            knee_x_traj.append(knee_x)
            knee_y_traj.append(knee_y)
            foot_y_traj.append(foot_y)

        # --- Body dimension constants ---
        foot_length = 0.12
        foot_height = 0.04
        joint_radius = 0.035
        mass_radius = 0.07       # point mass at hip
        seg_lw = 8               # limb line width

        # Colours
        limb_color = '#2C5F7C'      # darker blue for limbs
        joint_color = '#1B3A4B'     # dark joints
        mass_color = '#1B3A4B'      # point mass
        foot_color = '#1B3A4B'
        ground_color = '#8B7355'    # earthy brown

        # --- Figure setup: 1920x1080 (Full HD) white background ---
        dpi = 100
        fig_anim = plt.figure(figsize=(1920 / dpi, 1080 / dpi),
                              facecolor='white', dpi=dpi)
        ax_anim = fig_anim.add_subplot(111)
        ax_anim.set_facecolor('white')

        ax_anim.set_xlim(-1.5, 1.5)
        ax_anim.set_ylim(-0.15, 1.8)
        ax_anim.set_aspect('equal')
        ax_anim.axis('off')

        # Fill the figure with the axes (minimize whitespace)
        fig_anim.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Ground
        ground_rect = patches.Rectangle((-1.0, -0.15), 2.0, 0.15,
                                        facecolor=ground_color, edgecolor='none')
        ax_anim.add_patch(ground_rect)
        ax_anim.plot([-1.0, 1.0], [0, 0], color='#5C4033', lw=2)  # ground line

        # --- Create artist objects (initially empty) ---
        # Shadow ellipse on the ground
        shadow = patches.Ellipse((0, 0.005), 0.3, 0.03,
                                 facecolor='#00000020', edgecolor='none')
        ax_anim.add_patch(shadow)

        # Leg segments (solid rounded caps)
        thigh_line, = ax_anim.plot([], [], color=limb_color, lw=seg_lw,
                                   solid_capstyle='round', zorder=3)
        shank_line, = ax_anim.plot([], [], color=limb_color, lw=seg_lw,
                                   solid_capstyle='round', zorder=3)

        # Joint circles (knee, ankle)
        knee_joint = plt.Circle((0, 0), joint_radius, fc=joint_color, ec='none', zorder=4)
        ankle_joint = plt.Circle((0, 0), joint_radius * 0.8, fc=joint_color, ec='none', zorder=4)
        ax_anim.add_patch(knee_joint)
        ax_anim.add_patch(ankle_joint)

        # Foot
        foot_patch = patches.FancyBboxPatch((0, 0), foot_length, foot_height,
                                            boxstyle="round,pad=0.01",
                                            facecolor=foot_color, edgecolor='none', zorder=3)
        ax_anim.add_patch(foot_patch)

        # Point mass at hip
        mass_circle = plt.Circle((0, 0), mass_radius, fc=mass_color, ec='none', zorder=5)
        ax_anim.add_patch(mass_circle)

        # Info text
        time_text = ax_anim.text(0.02, 0.97, '', transform=ax_anim.transAxes,
                                 fontsize=10, verticalalignment='top', family='monospace',
                                 bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                                           edgecolor='#cccccc', alpha=0.9))

        # Phase badge
        phase_text = ax_anim.text(0.98, 0.97, '', transform=ax_anim.transAxes,
                                  fontsize=11, fontweight='bold', verticalalignment='top',
                                  horizontalalignment='right',
                                  bbox=dict(boxstyle='round,pad=0.3', facecolor='#4CAF50',
                                            edgecolor='none', alpha=0.9),
                                  color='white')

        # Frame indices
        indices = np.arange(0, len(results['time']), skip_frames)

        def update_anim(frame):
            idx = indices[frame] if frame < len(indices) else indices[-1]
            if idx >= len(results['time']):
                idx = len(results['time']) - 1

            y_hip = results['height'][idx]
            foot_y = foot_y_traj[idx]
            knee_y = knee_y_traj[idx]
            knee_x = knee_x_traj[idx]
            is_stance = y_hip <= self.skeleton.l_f

            # --- Update leg segments ---
            shank_line.set_data([0, knee_x], [foot_y, knee_y])
            thigh_line.set_data([knee_x, 0], [knee_y, y_hip])

            # --- Update joints ---
            knee_joint.set_center((knee_x, knee_y))
            ankle_joint.set_center((0, foot_y))

            # --- Foot ---
            foot_patch.set_x(-foot_length * 0.4)
            foot_patch.set_y(foot_y - foot_height)

            # --- Point mass at hip ---
            mass_circle.set_center((0, y_hip))

            # --- Shadow on ground ---
            shadow_scale = max(0.1, 0.35 - 0.15 * (y_hip - 0.8))
            shadow.set_width(shadow_scale)
            shadow_alpha = max(0.04, 0.12 - 0.06 * (y_hip - 0.8))
            shadow.set_facecolor(f'#000000{int(shadow_alpha * 255):02x}')

            # --- Phase badge ---
            if is_stance:
                phase_text.set_text('STANCE')
                phase_text.get_bbox_patch().set_facecolor('#FF7043')
            else:
                phase_text.set_text('FLIGHT')
                phase_text.get_bbox_patch().set_facecolor('#4CAF50')

            # --- Info text ---
            time_text.set_text(
                f't = {results["time"][idx]:.3f} s\n'
                f'h = {y_hip:.3f} m\n'
                f'v = {results["velocity"][idx]:.2f} m/s\n'
                f'F = {results["leg_force"][idx]:.0f} N')

            return (thigh_line, shank_line, knee_joint, ankle_joint,
                    foot_patch, mass_circle, shadow, time_text, phase_text)

        # Create animation
        n_frames = len(indices)
        ani = animation.FuncAnimation(fig_anim, update_anim, frames=n_frames,
                                      interval=10, blit=True, repeat=True)



        if self.save_video:
            # Save to Videos/ folder in the project directory
            project_dir = os.path.dirname(os.path.abspath(__file__))
            videos_dir = os.path.join(project_dir, 'Videos')
            os.makedirs(videos_dir, exist_ok=True)

            # Save mp4
            video_path = os.path.join(videos_dir, 'hopping_animation.mp4')
            print(f"Saving animation to {video_path}...")
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=60, metadata=dict(artist='Simulation'), bitrate=5000)
            ani.save(video_path, writer=writer, dpi=dpi)
            print(f"Animation saved successfully to {video_path}")

            # Convert to gif
            gif_path = os.path.join(videos_dir, 'hopping_animation.gif')
            print(f"Saving GIF to {gif_path}...")
            ani.save(gif_path, writer='pillow', fps=30, dpi=dpi)
            print(f"GIF saved successfully to {gif_path}")

            plt.close(fig_anim)
        else:
            plt.show()
            plt.close(fig_anim)

        return ani


        

    
