import csv
import datetime
import json
import math
import os
import random
import sys
import threading
import time
from collections import deque

import bpy
import matplotlib.pyplot as plt
import numpy as np
from mathutils import Vector
import networkx as nx
from scipy.optimize import minimize

from utils import calculate_world_bbox, get_object_data


class BlenderDTVisualizer:
    """Digital Twin visualizer using Blender instead of Matplotlib."""

    def __init__(self, digital_twin, logger, blend_file_path="assets/scene.blend"):
        self.digital_twin = digital_twin
        self.logger = logger
        self.blend_file_path = blend_file_path
        self.object_mapping = {}  # Maps DT object_ids to Blender object names
        self.initialized = False
        self.telemetry_history = {}  # Store telemetry history for plotting
        self.chair_position_history = {}  # Store chair position history

        # Default object mappings - adjust based on your actual Blender scene names
        self.object_type_prefixes = {
            "desk": "desk",
            "chair": "chair",
            "laptop": "laptop",
            "monitor": "monitor",
            "keyboard": "keyboard",
            "mouse": "mouse"
        }

        # Colors for different object types (RGBA)
        self.object_colors = {
            "desk": (0.6, 0.4, 0.2, 1.0),  # Brown
            "chair": (0.4, 0.4, 0.4, 1.0),  # Gray
            "laptop": (0.5, 0.8, 0.9, 1.0),  # Silver
            "monitor": (0.1, 0.1, 0.3, 1.0),  # Dark blue
            "keyboard": (0.2, 0.2, 0.2, 1.0),  # Dark gray
            "mouse": (0.7, 0.7, 0.7, 1.0)  # Light gray
        }

    def initialize(self):
        """Initialize Blender scene for visualization."""
        if self.initialized:
            return

        # Load the blend file if it's not the current open file
        current_path = bpy.data.filepath

        if os.path.abspath(current_path) != os.path.abspath(self.blend_file_path):
            print(f"Loading Blender file: {self.blend_file_path}")
            bpy.ops.wm.open_mainfile(filepath=self.blend_file_path)

        # Map digital twin objects to Blender objects
        self._map_objects()

        # Set up room dimensions
        self._setup_room()

        # Initialize telemetry history for plotting
        self._initialize_history_tracking()

        self.initialized = True
        print("Blender visualizer initialized")

    def _initialize_history_tracking(self):
        """Initialize history tracking for telemetry and chair positions."""
        # Track telemetry for laptops and monitors
        for obj_id, obj in self.digital_twin.objects.items():
            if obj.obj_type in ["laptop", "monitor"]:
                self.telemetry_history[obj_id] = {
                    "time": [],
                    "cpu_usage": [] if "cpu_usage" in obj.telemetry else None,
                    "memory_usage": [] if "memory_usage" in obj.telemetry else None,
                    "temperature": [] if "temperature" in obj.telemetry else None
                }

            # Track positions for chairs
            if obj.obj_type == "chair":
                self.chair_position_history[obj_id] = {
                    "time": [],
                    "x": [],
                    "y": [],
                    "occupied": [] if "occupied" in obj.telemetry else None
                }

    def _map_objects(self):
        """Map digital twin objects to Blender scene objects."""
        print("Mapping digital twin objects to Blender objects...")

        # Get all objects in the Blender scene
        blender_objects = bpy.data.objects
        print(f"Found {len(blender_objects)} objects in Blender scene")

        # Map each DT object to a Blender object
        for obj_id, dt_obj in self.digital_twin.objects.items():
            obj_type = dt_obj.obj_type

            # Find a corresponding Blender object based on type
            prefix = self.object_type_prefixes.get(obj_type)
            if not prefix:
                print(f"Warning: No prefix mapping for object type '{obj_type}'")
                continue

            # Try to find an object with matching type in the name
            matching_obj = None

            # First look for exact match with object_id
            for bl_obj in blender_objects:
                if bl_obj.name.lower() == obj_id.lower():
                    matching_obj = bl_obj
                    break

            # If no exact match, look for type prefix
            if not matching_obj:
                for bl_obj in blender_objects:
                    if prefix.lower() in bl_obj.name.lower() and bl_obj.name not in self.object_mapping.values():
                        matching_obj = bl_obj
                        break

            if matching_obj:
                self.object_mapping[obj_id] = matching_obj.name
                print(f"Mapped {obj_id} to Blender object '{matching_obj.name}'")

                # Update Blender object scale based on DT object dimensions
                dimensions = dt_obj.get_dimensions()

            else:
                print(f"Warning: Could not find a Blender object for {obj_id} of type {obj_type}")

        print(f"Mapped {len(self.object_mapping)} digital twin objects to Blender objects")

    def _setup_room(self):
        """Set up the room dimensions in Blender."""
        # Look for a room object in Blender
        room_obj = None
        for obj in bpy.data.objects:
            if "room" in obj.name.lower() or "floor" in obj.name.lower():
                room_obj = obj
                break

        if not room_obj:
            print("No room object found in Blender scene. Room dimensions won't be adjusted.")
            return

        # Get room dimensions from digital twin
        dt_room_dims = self.digital_twin.room_dimensions
        print(f"Setting room dimensions to {dt_room_dims}")

        # Scale the room object
        room_obj.scale.x = dt_room_dims["x"] / 2
        room_obj.scale.y = dt_room_dims["y"] / 2
        room_obj.scale.z = dt_room_dims["z"] / 2

    def update_scene(self):
        """Update Blender scene to match current digital twin state."""
        if not self.initialized:
            self.initialize()

        # Update object positions and properties
        for obj_id, dt_obj in self.digital_twin.objects.items():
            if obj_id in self.object_mapping:
                blender_obj_name = self.object_mapping[obj_id]
                blender_obj = bpy.data.objects.get(blender_obj_name)

                if blender_obj:
                    # Update position
                    blender_obj.location.x = dt_obj.position["x"]
                    blender_obj.location.y = dt_obj.position["y"]
                    blender_obj.location.z = dt_obj.position["z"]

                    # Update color/material based on telemetry
                    self._update_object_appearance(blender_obj, dt_obj)
                else:
                    print(f"Warning: Blender object '{blender_obj_name}' not found in scene")

            # Update history for plotting
            self._update_history_data(obj_id, dt_obj)

        # Update spatial relationship visualizations
        self._update_relationships()

    def _update_history_data(self, obj_id, dt_obj):
        """Update history data for telemetry and chair positions."""
        # Update telemetry history for laptops and monitors
        if obj_id in self.telemetry_history:
            self.telemetry_history[obj_id]["time"].append(self.digital_twin.simulation_time)

            # Only keep the last 50 data points
            if len(self.telemetry_history[obj_id]["time"]) > 50:
                self.telemetry_history[obj_id]["time"] = self.telemetry_history[obj_id]["time"][-50:]

            # Update each telemetry value if it exists
            for metric in ["cpu_usage", "memory_usage", "temperature"]:
                if self.telemetry_history[obj_id][metric] is not None:
                    if metric in dt_obj.telemetry:
                        self.telemetry_history[obj_id][metric].append(dt_obj.telemetry[metric])
                    else:
                        self.telemetry_history[obj_id][metric].append(None)

                    if len(self.telemetry_history[obj_id][metric]) > 50:
                        self.telemetry_history[obj_id][metric] = self.telemetry_history[obj_id][metric][-50:]

        # Update chair position history
        if obj_id in self.chair_position_history:
            self.chair_position_history[obj_id]["time"].append(self.digital_twin.simulation_time)
            self.chair_position_history[obj_id]["x"].append(dt_obj.position["x"])
            self.chair_position_history[obj_id]["y"].append(dt_obj.position["y"])

            # Keep only last 50 points
            if len(self.chair_position_history[obj_id]["time"]) > 50:
                self.chair_position_history[obj_id]["time"] = self.chair_position_history[obj_id]["time"][-50:]
                self.chair_position_history[obj_id]["x"] = self.chair_position_history[obj_id]["x"][-50:]
                self.chair_position_history[obj_id]["y"] = self.chair_position_history[obj_id]["y"][-50:]

            # Update occupied status if it exists
            if self.chair_position_history[obj_id]["occupied"] is not None:
                self.chair_position_history[obj_id]["occupied"].append(
                    1 if dt_obj.telemetry.get("occupied", False) else 0
                )

                if len(self.chair_position_history[obj_id]["occupied"]) > 50:
                    self.chair_position_history[obj_id]["occupied"] = self.chair_position_history[obj_id]["occupied"][
                                                                      -50:]

    def _update_object_appearance(self, blender_obj, dt_obj):
        """Update object appearance based on telemetry data."""
        obj_type = dt_obj.obj_type

        # Get base color for this object type
        base_color = self.object_colors.get(obj_type, (0.8, 0.8, 0.8, 1.0))

        # Choose or create a material
        mat_name = f"{dt_obj.object_id}_material"
        if mat_name in bpy.data.materials:
            mat = bpy.data.materials[mat_name]
        else:
            # Create new material
            mat = bpy.data.materials.new(name=mat_name)
            mat.use_nodes = True

            # Assign to object if not already assigned
            if blender_obj.data.materials:
                blender_obj.data.materials[0] = mat
            else:
                blender_obj.data.materials.append(mat)

        # Get the principled BSDF node
        nodes = mat.node_tree.nodes
        bsdf = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)

        if not bsdf:
            print(f"Warning: Could not find BSDF node in material {mat_name}")
            return

        # Modify the color based on telemetry
        color = list(base_color)

        # Object-specific appearance updates
        if obj_type == "laptop" or obj_type == "monitor":
            # Set color based on temperature if available
            if "temperature" in dt_obj.telemetry:
                temp = dt_obj.telemetry["temperature"]
                # Make it more blue when cool, more red when hot
                r_factor = min(1.0, max(0.0, (temp - 30) / 60))
                color[0] = base_color[0] + r_factor * 0.3  # More red with temp
                color[2] = base_color[2] * (1.0 - r_factor * 0.5)  # Less blue with temp

        elif obj_type == "chair":
            # Change color if occupied
            if "occupied" in dt_obj.telemetry and dt_obj.telemetry["occupied"]:
                color = (0.7, 0.1, 0.1, 1.0)  # Red when occupied

        # Apply the color
        bsdf.inputs['Base Color'].default_value = color

    def _update_relationships(self):
        """Visualize spatial relationships between objects."""
        # Clear any existing relationship visualizations
        for obj in bpy.data.objects:
            if obj.name.startswith("relation_"):
                bpy.data.objects.remove(obj)

        # Add new relationship visualizations for important relationships
        for obj_id, dt_obj in self.digital_twin.objects.items():
            if obj_id not in self.object_mapping:
                continue

            source_obj = bpy.data.objects[self.object_mapping[obj_id]]

            # Create visualizations for each relationship type
            for relation_type, related_ids in dt_obj.spatial_relations.items():
                # Only visualize certain relationships
                if relation_type not in ['on', 'in_front_of']:
                    continue

                for related_id in related_ids:
                    if related_id not in self.object_mapping:
                        continue

                    target_obj = bpy.data.objects[self.object_mapping[related_id]]

                    # Create a line between objects
                    self._create_relationship_line(
                        source_obj.location,
                        target_obj.location,
                        relation_type,
                        f"relation_{obj_id}_to_{related_id}_{relation_type}"
                    )

    def _create_relationship_line(self, start_loc, end_loc, relation_type, name):
        """Create a line visualizing a relationship between two objects."""
        # Define curve color and appearance based on relationship type
        if relation_type == 'on':
            curve_color = (0, 1, 0, 0.7)  # Green
            thickness = 0.02
        elif relation_type == 'in_front_of':
            curve_color = (0, 0, 1, 0.5)  # Blue
            thickness = 0.01
        else:
            curve_color = (0.5, 0.5, 0.5, 0.3)  # Gray
            thickness = 0.01

        # Create the curve object
        curve_data = bpy.data.curves.new(name=f"{name}_data", type='CURVE')
        curve_data.dimensions = '3D'
        curve_data.resolution_u = 2

        # Create the line
        polyline = curve_data.splines.new('POLY')
        polyline.points.add(1)  # Two points for start and end
        polyline.points[0].co = (start_loc.x, start_loc.y, start_loc.z, 1)
        polyline.points[1].co = (end_loc.x, end_loc.y, end_loc.z, 1)

        # Create curve object
        curve_obj = bpy.data.objects.new(name, curve_data)
        bpy.context.collection.objects.link(curve_obj)

        # Create and assign material
        mat = bpy.data.materials.new(name=f"{name}_material")
        mat.use_nodes = True

        # Set material properties
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        # Clear default nodes
        nodes.clear()

        # Create emission shader
        emission = nodes.new(type='ShaderNodeEmission')
        emission.inputs['Color'].default_value = curve_color
        emission.inputs['Strength'].default_value = 1.0

        # Create material output
        output = nodes.new(type='ShaderNodeOutputMaterial')

        # Link nodes
        links.new(emission.outputs['Emission'], output.inputs['Surface'])

        # Assign material to curve
        curve_data.materials.append(mat)

        # Set curve thickness
        curve_data.bevel_depth = thickness

    def create_visualization_frame(self, frame_number):
        """Create visualization frame for the current state using Blender."""
        # Update Blender scene to match current DT state
        self.update_scene()

        # Set render parameters
        scene = bpy.context.scene
        scene.render.image_settings.file_format = 'PNG'

        # Create output directory if it doesn't exist
        os.makedirs(self.logger.img_dir, exist_ok=True)

        # Set output path
        output_path = os.path.join(self.logger.img_dir, f"frame_{frame_number:04d}.png")
        scene.render.filepath = output_path

        # Add scene information as text overlay
        self._add_scene_info_overlay()

        # Create telemetry plots
        self._create_telemetry_plots(frame_number)

        # Create chair position plots
        self._create_chair_position_plots(frame_number)

        # Render the scene
        print(f"Rendering frame {frame_number} to {output_path}")
        bpy.ops.render.render(write_still=True)

        return output_path

    def _create_telemetry_plots(self, frame_number):
        """Create telemetry plots using matplotlib."""
        # Only proceed if we have telemetry data
        if not self.telemetry_history:
            return

        # Create a figure with subplots for each metric
        fig, axes = plt.subplots(3, 1, figsize=(8, 10), tight_layout=True)

        # Set titles for each subplot
        axes[0].set_title('CPU Usage (%)')
        axes[1].set_title('Memory Usage (%)')
        axes[2].set_title('Temperature (°C)')

        # Plot data for each object
        for obj_id, data in self.telemetry_history.items():
            time_data = data["time"]

            # Plot CPU usage if available
            if data["cpu_usage"] is not None and len(time_data) > 1:
                axes[0].plot(time_data, data["cpu_usage"], label=obj_id)
                axes[0].set_ylim(0, 100)

            # Plot memory usage if available
            if data["memory_usage"] is not None and len(time_data) > 1:
                axes[1].plot(time_data, data["memory_usage"], label=obj_id)
                axes[1].set_ylim(0, 100)

            # Plot temperature if available
            if data["temperature"] is not None and len(time_data) > 1:
                axes[2].plot(time_data, data["temperature"], label=obj_id)
                axes[2].set_ylim(30, 90)

        # Add labels and grid
        for ax in axes:
            ax.set_xlabel('Simulation Time (s)')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()

        # Save the figure
        telemetry_plot_path = os.path.join(self.logger.img_dir, f"telemetry_{frame_number:04d}.png")
        fig.savefig(telemetry_plot_path, dpi=100)
        plt.close(fig)

        return telemetry_plot_path

    def _create_chair_position_plots(self, frame_number):
        """Create chair position plots using matplotlib."""
        # Only proceed if we have chair position data
        if not self.chair_position_history:
            return

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)

        # Set titles
        ax1.set_title('Chair Movement (Top View)')
        ax2.set_title('Chair Occupancy Over Time')

        # Room boundaries for plotting
        room_x = self.digital_twin.room_dimensions["x"]
        room_y = self.digital_twin.room_dimensions["y"]

        # Define colors for each chair to maintain consistency
        chair_colors = {
            "chair1": "blue",
            "chair2": "green",
            "chair3": "purple"
        }

        # Plot chair positions on the first subplot
        for chair_id, data in self.chair_position_history.items():
            # Get the base color for this chair
            base_color = chair_colors.get(chair_id, "gray")

            # Only plot if we have data
            if data["x"] and data["y"]:
                # Get the last 10 points (or fewer if not enough data)
                trail_length = min(10, len(data["x"]))
                x_trail = data["x"][-trail_length:]
                y_trail = data["y"][-trail_length:]

                # Plot the full path lightly in the background (faded)
                if len(data["x"]) > 1:
                    ax1.plot(data["x"], data["y"], '-', linewidth=1, alpha=0.2, color=base_color)

                # Plot each segment of the trail with increasing alpha
                for i in range(trail_length - 1):
                    # Calculate alpha based on position in trail (more recent = more opaque)
                    alpha = 0.3 + (0.7 * i / (trail_length - 1))

                    # Plot the segment
                    ax1.plot(x_trail[i:i + 2], y_trail[i:i + 2], '-o',
                             linewidth=2,
                             markersize=4,
                             alpha=alpha,
                             color=base_color)

                # Highlight current position
                is_occupied = data["occupied"] and data["occupied"][-1] == 1

                # Mark the current position with a bigger marker
                ax1.plot(data["x"][-1], data["y"][-1], 'o',
                         markersize=10,
                         markeredgecolor='black',
                         markeredgewidth=1.5,
                         color='red' if is_occupied else base_color,
                         label=f"{chair_id}" + (" (occupied)" if is_occupied else ""))

                # Add a text label for the chair
                ax1.annotate(chair_id,
                             (data["x"][-1], data["y"][-1]),
                             xytext=(5, 5),
                             textcoords='offset points',
                             fontsize=9,
                             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        # Add desk positions for reference (if available)
        for obj_id, obj in self.digital_twin.objects.items():
            if obj.obj_type == "desk":
                ax1.plot(obj.position["x"], obj.position["y"], 's',
                         markersize=8, color='brown', alpha=0.6)
                ax1.annotate(obj_id,
                             (obj.position["x"], obj.position["y"]),
                             xytext=(0, -15),
                             textcoords='offset points',
                             fontsize=8,
                             ha='center',
                             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7))

        # Set room boundaries and grid
        ax1.set_xlim(-room_x, room_x)
        ax1.set_ylim(-room_y, room_y)
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.legend(loc='upper right')

        # Plot chair occupancy on the second subplot with distinct colors
        for chair_id, data in self.chair_position_history.items():
            if data["occupied"] is not None and len(data["time"]) > 1:
                ax2.step(data["time"], data["occupied"],
                         label=chair_id,
                         color=chair_colors.get(chair_id, "gray"),
                         linewidth=2)

                # Add markers at the transition points
                for i in range(1, len(data["occupied"])):
                    if data["occupied"][i] != data["occupied"][i - 1]:
                        ax2.plot(data["time"][i], data["occupied"][i], 'o',
                                 markersize=6,
                                 color=chair_colors.get(chair_id, "gray"))

        # Set y-axis limits and labels for occupancy
        ax2.set_ylim(-0.1, 1.1)
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['Empty', 'Occupied'])
        ax2.set_xlabel('Simulation Time (s)')
        ax2.set_ylabel('Chair Status')
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend(loc='upper right')

        # Add a timestamp to the figure
        fig.suptitle(f'Chair Analysis - Simulation Time: {self.digital_twin.simulation_time}s',
                     fontsize=12, y=0.98)

        # Save the figure
        chair_plot_path = os.path.join(self.logger.img_dir, f"chairs_{frame_number:04d}.png")
        fig.savefig(chair_plot_path, dpi=100)
        plt.close(fig)

        return chair_plot_path

    def _add_scene_info_overlay(self):
        """Add scene information as text overlay in Blender."""
        # Clear existing text objects
        for obj in bpy.data.objects:
            if obj.type == 'FONT' and obj.name.startswith('info_'):
                bpy.data.objects.remove(obj)

        # Add simulation time text
        self._add_text_object(
            f"Digital Twin Simulation - Time: {self.digital_twin.simulation_time}s",
            "info_title",
            location=Vector((0, 2.5, 2.3)),
            size=0.15
        )

        # Add scene description text
        scene_desc = self.digital_twin.get_scene_description()
        lines = self._wrap_text(scene_desc, max_chars=60)

        for i, line in enumerate(lines[:5]):  # Limit to 5 lines
            self._add_text_object(
                line,
                f"info_desc_{i}",
                location=Vector((2.2, 2.0 - 0.15 * i, 2.0)),
                size=0.08
            )

        # Add telemetry values for objects
        y_pos = 2.0
        for obj_id, dt_obj in self.digital_twin.objects.items():
            if dt_obj.obj_type in ["laptop", "monitor"]:
                y_pos -= 0.15
                telemetry_text = f"{obj_id}: "

                if "cpu_usage" in dt_obj.telemetry:
                    telemetry_text += f"CPU {dt_obj.telemetry['cpu_usage']:.1f}% "

                if "memory_usage" in dt_obj.telemetry:
                    telemetry_text += f"Mem {dt_obj.telemetry['memory_usage']:.1f}% "

                if "temperature" in dt_obj.telemetry:
                    telemetry_text += f"Temp {dt_obj.telemetry['temperature']:.1f}°C"

                self._add_text_object(
                    telemetry_text,
                    f"info_telemetry_{obj_id}",
                    location=Vector((-2.5, y_pos, 2.0)),
                    size=0.08
                )

    def _add_text_object(self, text, name, location=Vector((0, 0, 0)), size=0.1, color=(1, 1, 1, 1)):
        """Add a text object to the Blender scene."""
        font_curve = bpy.data.curves.new(type="FONT", name=f"{name}_data")
        font_curve.body = text

        text_obj = bpy.data.objects.new(name, font_curve)
        text_obj.location = location
        text_obj.data.size = size

        # Create material for text
        mat = bpy.data.materials.new(name=f"{name}_material")
        mat.use_nodes = True

        # Set material properties
        nodes = mat.node_tree.nodes
        bsdf = nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Emission Strength"].default_value = 1.0

        # Assign material to text
        text_obj.data.materials.append(mat)

        # Add to scene
        bpy.context.collection.objects.link(text_obj)

        # Always face the camera
        text_obj.rotation_euler = (math.pi / 2, 0, 0)

        return text_obj

    def _wrap_text(self, text, max_chars=60):
        """Wrap text to fit within max_chars per line."""
        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            if len(current_line) + len(word) + 1 <= max_chars:
                if current_line:
                    current_line += " " + word
                else:
                    current_line = word
            else:
                lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        return lines


class DTLogger:
    """Logger for Digital Twin data."""

    def __init__(self, digital_twin, log_dir="dt_logs"):
        self.digital_twin = digital_twin
        self.log_dir = log_dir
        self.run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.log_dir, self.run_id)
        self.img_dir = os.path.join(self.run_dir, "images")
        self.data_dir = os.path.join(self.run_dir, "data")
        self.graph_dir = os.path.join(self.run_dir, "graph")
        self.csv_files = {}
        self.scene_desc_file = None

        # Create log directories
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.graph_dir, exist_ok=True)

        # Create scene description log file
        scene_file_path = os.path.join(self.data_dir, "scene_descriptions.csv")
        self.scene_desc_file = open(scene_file_path, 'w', newline='')
        scene_csv_writer = csv.writer(self.scene_desc_file)
        scene_csv_writer.writerow(["frame", "time", "scene_description"])
        self.scene_writer = scene_csv_writer

    def log_state(self, frame_number):
        """Log the current state of the digital twin."""
        # Get current state
        state = self.digital_twin.get_current_state()

        # Save state to JSON file
        state_file = os.path.join(self.data_dir, f"state_{frame_number:04d}.json")
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)

        # Log scene description
        scene_description = self.digital_twin.get_scene_description()
        self.scene_writer.writerow([
            frame_number,
            self.digital_twin.simulation_time,
            scene_description
        ])

        # Log telemetry to CSV files
        for obj_id, obj_data in state.items():
            # Skip the scene_description entry since it's not an object
            if obj_id == "scene_description":
                continue

            obj_type = obj_data["type"]

            # Create CSV file for this object type if it doesn't exist
            if obj_type not in self.csv_files:
                csv_path = os.path.join(self.data_dir, f"{obj_type}_telemetry.csv")

                # Get all telemetry keys for this object type
                telemetry_keys = list(obj_data["telemetry"].keys()) if "telemetry" in obj_data else []

                # Create CSV header
                header = ["frame", "time", "object_id"] + telemetry_keys

                # Create and store the CSV file
                csv_file = open(csv_path, 'w', newline='')
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(header)

                self.csv_files[obj_type] = {
                    "file": csv_file,
                    "writer": csv_writer,
                    "telemetry_keys": telemetry_keys
                }

            # Write telemetry data to CSV if telemetry exists
            if "telemetry" in obj_data:
                csv_info = self.csv_files[obj_type]
                telemetry_values = [obj_data["telemetry"].get(key, "") for key in csv_info["telemetry_keys"]]
                csv_info["writer"].writerow([
                                                frame_number,
                                                self.digital_twin.simulation_time,
                                                obj_id
                                            ] + telemetry_values)

        return state_file

    def close(self):
        """Close all open files."""
        for obj_type, csv_info in self.csv_files.items():
            csv_info["file"].close()

        if self.scene_desc_file:
            self.scene_desc_file.close()

        print(f"Logs saved to {self.run_dir}")


class DTObject:
    """Digital Twin object with both physical and functional properties."""

    def __init__(self, object_id, obj_type, position=None, properties=None):
        self.object_id = object_id
        self.obj_type = obj_type
        self.position = position or {"x": 0, "y": 0, "z": 0}
        self.status = "active"
        self.properties = properties or {}
        self.telemetry = {}
        self.telemetry_history = {}
        self.metrics = {}
        self.spatial_relations = {}  # New field for spatial relationships

        # Initialize type-specific properties and telemetry
        self._initialize_type_specific()

    def _initialize_type_specific(self):
        """Initialize properties and telemetry based on object type."""
        if self.obj_type == "laptop":
            # Default laptop properties if not provided
            if "cpu_cores" not in self.properties:
                self.properties["cpu_cores"] = 4
            if "ram_gb" not in self.properties:
                self.properties["ram_gb"] = 16
            if "storage_gb" not in self.properties:
                self.properties["storage_gb"] = 512
            # Ensure dimensions are set
            if "dimensions" not in self.properties:
                self.properties["dimensions"] = (0.35, 0.25, 0.03)

            # Initialize telemetry values
            self.telemetry = {
                "cpu_usage": 10.0,  # Start with non-zero values
                "memory_usage": 15.0,
                "temperature": 40.0,
                "power_state": "on"
            }

            # Initialize history for each telemetry metric
            for key in self.telemetry:
                self.telemetry_history[key] = deque(maxlen=100)  # Store last 100 values
                self.telemetry_history[key].append((0, self.telemetry[key]))  # Add initial value

        elif self.obj_type == "monitor":
            # Default monitor properties
            if "dimensions" not in self.properties:
                self.properties["dimensions"] = (0.6, 0.2, 0.4)

            # Monitor telemetry
            self.telemetry = {
                "temperature": 35.0,
                "power_state": "on",
                "cpu_usage": 15.0,  # Start with non-zero values
                "memory_usage": 20.0,
            }

            # Initialize history
            for key in self.telemetry:
                self.telemetry_history[key] = deque(maxlen=100)
                self.telemetry_history[key].append((0, self.telemetry[key]))

        elif self.obj_type == "chair":
            # Ensure dimensions are set
            if "dimensions" not in self.properties:
                self.properties["dimensions"] = (0.5, 0.5, 1.0)

            # Chair telemetry
            self.telemetry = {
                "occupied": False,
                "position_x": self.position["x"],
                "position_y": self.position["y"]
            }

            # Initialize history
            for key in self.telemetry:
                self.telemetry_history[key] = deque(maxlen=100)
                self.telemetry_history[key].append((0, self.telemetry[key]))  # Add initial value

        elif self.obj_type == "desk":
            # Ensure dimensions are set
            if "dimensions" not in self.properties:
                self.properties["dimensions"] = (1.2, 0.6, 0.75)

        elif self.obj_type == "keyboard":
            # Ensure dimensions are set
            if "dimensions" not in self.properties:
                self.properties["dimensions"] = (0.4, 0.15, 0.03)

        elif self.obj_type == "mouse":
            # Ensure dimensions are set
            if "dimensions" not in self.properties:
                self.properties["dimensions"] = (0.1, 0.06, 0.03)

    def get_dimensions(self):
        """Get the dimensions of the object."""
        return self.properties.get("dimensions", None)

    def update_position(self, new_position):
        """Update object position."""
        self.position = new_position

        # Update position telemetry if applicable
        if "position_x" in self.telemetry:
            self.telemetry["position_x"] = new_position["x"]
            self.telemetry_history["position_x"].append((time.time(), new_position["x"]))

        if "position_y" in self.telemetry:
            self.telemetry["position_y"] = new_position["y"]
            self.telemetry_history["position_y"].append((time.time(), new_position["y"]))

    def update_telemetry(self, telemetry_name, value):
        """Update a telemetry value and store in history."""
        if telemetry_name in self.telemetry:
            self.telemetry[telemetry_name] = value

            # Create history deque if it doesn't exist
            if telemetry_name not in self.telemetry_history:
                self.telemetry_history[telemetry_name] = deque(maxlen=100)

            self.telemetry_history[telemetry_name].append((time.time(), value))

    def simulate_telemetry_update(self):
        """Simulate changes in telemetry based on object type."""
        if self.obj_type == "laptop" or self.obj_type == "monitor":
            # Realistic CPU usage pattern (oscillating with random noise)
            base_cpu = random.uniform(10, 45) + 15 * np.sin(time.time() / 10)  # Base oscillation with faster period for demo
            cpu_usage = min(100, max(20, base_cpu + random.uniform(-5, 15)))  # Add noise, ensure min 5%

            # Memory usage (slowly increasing, occasional drops)
            mem_trend = 0.5 if random.random() > 0.9 else 0.1  # Faster changes for demo
            mem_reset = random.random() > 0.95  # Occasional reset (like app closing)
            current_mem = self.telemetry["memory_usage"]

            if mem_reset:
                memory_usage = max(10, current_mem * 0.7)  # Big drop
            else:
                memory_usage = min(95, current_mem + mem_trend)  # Slow increase

            # Temperature (correlates with CPU)
            base_temp = random.uniform(10, 40) + (cpu_usage / 100) * 20  # CPU impacts temperature
            temperature = min(90, max(48, base_temp + random.uniform(-0.5, 0.5)))

            # Update all telemetry
            self.update_telemetry("cpu_usage", cpu_usage)
            self.update_telemetry("memory_usage", memory_usage)
            self.update_telemetry("temperature", temperature)

            # Compute aggregate metrics
            self.metrics["system_load"] = (cpu_usage * 0.5 + memory_usage * 0.3 + temperature * 0.2) / 100
            self.metrics["health_score"] = 100 - (temperature - 35) - (memory_usage / 5)

        elif self.obj_type == "chair":
            # Randomly change occupied status - more frequent for demo
            if random.random() > 0.9:  # 10% chance on each update
                self.update_telemetry("occupied", not self.telemetry["occupied"])

    def get_telemetry_history(self, metric_name, count=None):
        """Get historical values for a specific telemetry metric."""
        if metric_name not in self.telemetry_history:
            return []

        history = list(self.telemetry_history[metric_name])
        if count:
            return history[-count:]
        return history

    # New methods for spatial relationships
    def add_spatial_relation(self, relation_type, target_object_id):
        """Add a spatial relationship to another object."""
        if relation_type not in self.spatial_relations:
            self.spatial_relations[relation_type] = []
        if target_object_id not in self.spatial_relations[relation_type]:
            self.spatial_relations[relation_type].append(target_object_id)

    def remove_spatial_relation(self, relation_type, target_object_id):
        """Remove a spatial relationship to another object."""
        if relation_type in self.spatial_relations and target_object_id in self.spatial_relations[relation_type]:
            self.spatial_relations[relation_type].remove(target_object_id)

    def to_dict(self):
        """Convert object to dictionary representation."""
        return {
            "id": self.object_id,
            "type": self.obj_type,
            "position": self.position,
            "status": self.status,
            "properties": self.properties,
            "telemetry": self.telemetry,
            "metrics": self.metrics,
            "spatial_relations": self.spatial_relations
        }


class SimpleDTEnvironment:
    """Digital Twin Environment with enhanced physical and functional aspects."""

    def __init__(self):
        self.objects = {}
        self.room_dimensions = {"x": 6, "y": 5, "z": 2.5}  # Default room size in meters
        self.history = []  # Store state history
        self.running = False
        self.simulation_thread = None
        self.simulation_time = 0
        self.spatial_relation_thresholds = {
            "on": 0.1,  # Vertical distance threshold for 'on' relation
            "under": 0.1,  # Vertical distance threshold for 'under' relation
            "proximity": 1.0  # Horizontal distance threshold for proximity relations
        }

    def set_room_dimensions(self, x, y, z):
        """Set room dimensions."""
        self.room_dimensions = {"x": x, "y": y, "z": z}

    def add_object(self, object_id, obj_type, position=None, properties=None):
        """Add a new object to the digital twin."""
        self.objects[object_id] = DTObject(object_id, obj_type, position, properties)

        # Update spatial relationships for new object
        self.update_spatial_relationships(object_id)

        return self.objects[object_id]

    def update_object_position(self, object_id, new_position):
        """Update position of an object."""
        if object_id in self.objects:
            # First check if the new position would cause overlap
            # if self.check_object_overlap(object_id, new_position):
            #     # If there's overlap, don't update the position
            #     return False
            #
            # # Check if the position is within room boundaries
            # if not self._is_position_in_room(new_position):
            #     return False

            # If no overlap and within room, proceed with position update
            self.objects[object_id].update_position(new_position)

            # Update spatial relationships after position change
            self.update_spatial_relationships(object_id)

            return True

        return False

    def _is_position_in_room(self, position):
        """Check if a position is within the room boundaries."""
        # Get room dimensions
        room_x = self.room_dimensions["x"]
        room_y = self.room_dimensions["y"]
        room_z = self.room_dimensions["z"]

        # Check if position is within room boundaries
        return (
                -room_x / 2 <= position["x"] <= room_x / 2 and
                -room_y / 2 <= position["y"] <= room_y / 2 and
                0 <= position["z"] <= room_z
        )

    def add_spatial_relation(self, source_id, relation_type, target_id):
        """Add a spatial relationship between two objects."""
        if source_id in self.objects and target_id in self.objects:
            self.objects[source_id].add_spatial_relation(relation_type, target_id)
            return True
        return False

    def remove_spatial_relation(self, source_id, relation_type, target_id):
        """Remove a spatial relationship between two objects."""
        if source_id in self.objects and target_id in self.objects:
            self.objects[source_id].remove_spatial_relation(relation_type, target_id)
            return True
        return False

    def update_spatial_relationships(self, object_id=None):
        """Update spatial relationships based on physical positions and bounding boxes from Blender data."""
        with open("assets/scene_info.json", "r") as f:
            scene_info = json.load(f)
        # If no object_id is provided, update all objects
        objects_to_update = [object_id] if object_id else list(self.objects.keys())

        for obj_id in objects_to_update:
            # print('obj_id', obj_id)
            if obj_id not in self.objects:
                continue

            obj = self.objects[obj_id]

            # Clear existing positional relationships (keeping others like functional relationships)
            positional_relations = ['on', 'under', 'in_front_of', 'behind', 'adjacent_to']
            for rel in positional_relations:
                if rel in obj.spatial_relations:
                    obj.spatial_relations[rel] = []

            # Get obj's blender data if available
            obj_blender_data = get_object_data(scene_info, obj_id)
            if not obj_blender_data:
                continue  # Skip if we don't have Blender data

            # Calculate obj's bounding box in world space
            # [min_x, min_y, min_z, max_x, max_y, max_z]
            obj_bbox = calculate_world_bbox(obj_blender_data)
            # print('obj_bbox', obj_bbox)
            if not obj_bbox:
                continue

            # Check relationships with other objects
            for other_id, other in self.objects.items():
                if other_id == obj_id:
                    continue

                # Get other's blender data
                other_blender_data = get_object_data(scene_info, other_id)
                if not other_blender_data:
                    continue

                # Calculate other's bounding box in world space
                other_bbox = calculate_world_bbox(other_blender_data)
                if not other_bbox:
                    continue

                # accurate bounding boxes for both objects in world space to determine their spatial relationships

                # Bounding box format: [min_x, min_y, min_z, max_x, max_y, max_z]

                # Check if one object is above the other (with horizontal overlap)
                horizontal_overlap = (
                        obj_bbox[0] < other_bbox[3] and  # obj's min_x < other's max_x
                        obj_bbox[3] > other_bbox[0] and  # obj's max_x > other's min_x
                        obj_bbox[1] < other_bbox[4] and  # obj's min_y < other's max_y
                        obj_bbox[4] > other_bbox[1]  # obj's max_y > other's min_y
                )

                # 'on' relationship: obj's bottom is close to other's top with horizontal overlap
                on_threshold = self.spatial_relation_thresholds.get("on", 0.05)  # Adjust threshold as needed
                if (horizontal_overlap and
                        abs(obj_bbox[2] - other_bbox[5]) <= on_threshold and
                        obj_bbox[2] >= other_bbox[5]):  # obj's min_z >= other's max_z
                    obj.add_spatial_relation('on', other_id)
                    other.add_spatial_relation('has_on_top', obj_id)

                # 'under' relationship: obj's top is close to other's bottom with horizontal overlap
                under_threshold = self.spatial_relation_thresholds.get("under", 0.05)
                if (horizontal_overlap and
                        abs(obj_bbox[5] - other_bbox[2]) <= under_threshold and
                        obj_bbox[5] <= other_bbox[2]):  # obj's max_z <= other's min_z
                    obj.add_spatial_relation('under', other_id)
                    other.add_spatial_relation('above', obj_id)

                # Vertical overlap check for front/behind relationships
                vertical_overlap = (
                        obj_bbox[2] < other_bbox[5] and  # obj's min_z < other's max_z
                        obj_bbox[5] > other_bbox[2]  # obj's max_z > other's min_z
                )

                # x-axis overlap for front/behind relationships
                x_axis_overlap = (
                        obj_bbox[0] < other_bbox[3] and  # obj's min_x < other's max_x
                        obj_bbox[3] > other_bbox[0]  # obj's max_x > other's min_x
                )

                # 'in_front_of' relationship (assuming +y means "forward")
                # obj is in front of other if obj's min_y > other's max_y with x and z overlap
                if vertical_overlap and x_axis_overlap and obj_bbox[1] > other_bbox[4]:
                    obj.add_spatial_relation('in_front_of', other_id)
                    other.add_spatial_relation('behind', obj_id)

                # 'behind' relationship (assuming +y means "forward")
                # obj is behind other if obj's max_y < other's min_y with x and z overlap
                if vertical_overlap and x_axis_overlap and obj_bbox[4] < other_bbox[1]:
                    obj.add_spatial_relation('behind', other_id)
                    other.add_spatial_relation('in_front_of', obj_id)

                # 'adjacent_to' relationship
                # Objects are adjacent if they're close horizontally but not on/under each other
                proximity_threshold = self.spatial_relation_thresholds.get("proximity", 0.3)

                # Calculate the shortest distance between the two bounding boxes
                if obj_bbox[3] < other_bbox[0]:  # obj is entirely to the left of other
                    x_distance = other_bbox[0] - obj_bbox[3]
                elif other_bbox[3] < obj_bbox[0]:  # other is entirely to the left of obj
                    x_distance = obj_bbox[0] - other_bbox[3]
                else:  # x-overlap
                    x_distance = 0

                if obj_bbox[4] < other_bbox[1]:  # obj is entirely behind other
                    y_distance = other_bbox[1] - obj_bbox[4]
                elif other_bbox[4] < obj_bbox[1]:  # other is entirely behind obj
                    y_distance = obj_bbox[1] - other_bbox[4]
                else:  # y-overlap
                    y_distance = 0

                # Horizontal distance between bounding boxes
                horizontal_distance = math.sqrt(x_distance ** 2 + y_distance ** 2)

                if (horizontal_distance <= proximity_threshold and
                        vertical_overlap and
                        not (abs(obj_bbox[2] - other_bbox[5]) <= on_threshold or
                             abs(obj_bbox[5] - other_bbox[2]) <= under_threshold)):
                    obj.add_spatial_relation('adjacent_to', other_id)
                    # adjacent_to is symmetric
                    other.add_spatial_relation('adjacent_to', obj_id)

    def check_object_overlap(self, obj_id, position):
        """
        Check if the proposed position would cause the object to overlap with other objects.
        Returns True if there's overlap, False otherwise.
        """
        if obj_id not in self.objects:
            return False

        obj = self.objects[obj_id]
        obj_dimensions = obj.get_dimensions()

        if not obj_dimensions:
            # If dimensions aren't available, assume no overlap
            return False

        # Calculate object bounds based on center position and dimensions
        obj_half_width = obj_dimensions[0] / 2
        obj_half_depth = obj_dimensions[1] / 2
        obj_height = obj_dimensions[2]

        obj_min_x = position["x"] - obj_half_width
        obj_max_x = position["x"] + obj_half_width
        obj_min_y = position["y"] - obj_half_depth
        obj_max_y = position["y"] + obj_half_depth
        obj_min_z = position["z"]
        obj_max_z = position["z"] + obj_height

        # Check overlap with each other object
        for other_id, other in self.objects.items():
            if other_id == obj_id:
                continue

            # Get other object dimensions and position
            other_dimensions = other.get_dimensions()
            if not other_dimensions:
                continue  # Skip if dimensions aren't available

            other_half_width = other_dimensions[0] / 2
            other_half_depth = other_dimensions[1] / 2
            other_height = other_dimensions[2]

            other_pos = other.position

            # Calculate other object bounds
            other_min_x = other_pos["x"] - other_half_width
            other_max_x = other_pos["x"] + other_half_width
            other_min_y = other_pos["y"] - other_half_depth
            other_max_y = other_pos["y"] + other_half_depth
            other_min_z = other_pos["z"]
            other_max_z = other_pos["z"] + other_height

            # Check for overlap in all dimensions
            if (obj_min_x < other_max_x and obj_max_x > other_min_x and
                    obj_min_y < other_max_y and obj_max_y > other_min_y and
                    obj_min_z < other_max_z and obj_max_z > other_min_z):
                return True

        return False

    def get_scene_description(self):
        """Generate a natural language description of the scene based on spatial relationships."""
        descriptions = []

        # First, identify key objects by type
        desks = [obj for obj_id, obj in self.objects.items() if obj.obj_type == "desk"]
        computers = [obj for obj_id, obj in self.objects.items() if obj.obj_type in ["laptop", "monitor"]]
        chairs = [obj for obj_id, obj in self.objects.items() if obj.obj_type == "chair"]
        input_devices = [obj for obj_id, obj in self.objects.items() if obj.obj_type in ["keyboard", "mouse"]]

        # Describe objects on desks
        for desk in desks:
            items_on_desk = []

            # Find all objects that have an 'on' relationship with this desk
            for obj in self.objects.values():
                if 'on' in obj.spatial_relations and desk.object_id in obj.spatial_relations['on']:
                    items_on_desk.append(obj)

            if items_on_desk:
                item_descriptions = [f"{obj.obj_type} ({obj.object_id})" for obj in items_on_desk]
                if len(item_descriptions) == 1:
                    descriptions.append(f"There is a {item_descriptions[0]} on the {desk.object_id}.")
                elif len(item_descriptions) == 2:
                    descriptions.append(
                        f"There are a {item_descriptions[0]} and a {item_descriptions[1]} on the {desk.object_id}.")
                else:
                    items_str = ", ".join(item_descriptions[:-1]) + f", and {item_descriptions[-1]}"
                    descriptions.append(f"There are {items_str} on the {desk.object_id}.")

        # Describe chair positions
        for chair in chairs:
            # Find desks this chair is in front of
            in_front_of_desks = []
            for desk in desks:
                if 'in_front_of' in chair.spatial_relations and desk.object_id in chair.spatial_relations[
                    'in_front_of']:
                    in_front_of_desks.append(desk)

            if in_front_of_desks:
                for desk in in_front_of_desks:
                    descriptions.append(f"The {chair.object_id} is positioned in front of the {desk.object_id}.")

                # Add occupancy information
                if chair.telemetry.get("occupied", False):
                    descriptions.append(f"The {chair.object_id} is currently occupied.")
            else:
                descriptions.append(f"There is a {chair.object_id} in the room.")

        # Describe computer telemetry
        for computer in computers:
            if computer.obj_type == "laptop":
                telemetry_desc = []
                if "cpu_usage" in computer.telemetry:
                    cpu_usage = computer.telemetry["cpu_usage"]
                    if cpu_usage > 80:
                        telemetry_desc.append("high CPU usage")
                    elif cpu_usage > 50:
                        telemetry_desc.append("moderate CPU usage")

                if "temperature" in computer.telemetry:
                    temp = computer.telemetry["temperature"]
                    if temp > 75:
                        telemetry_desc.append("high temperature")
                    elif temp > 60:
                        telemetry_desc.append("elevated temperature")

                if telemetry_desc:
                    descriptions.append(f"The {computer.object_id} is showing {' and '.join(telemetry_desc)}.")

        # Describe input device relationships (keyboards and mice)
        for device in input_devices:
            adjacent_computers = []
            for computer in computers:
                if 'adjacent_to' in device.spatial_relations and computer.object_id in device.spatial_relations[
                    'adjacent_to']:
                    adjacent_computers.append(computer)

            if adjacent_computers:
                for computer in adjacent_computers:
                    descriptions.append(f"The {device.object_id} is next to the {computer.object_id}.")

        # Create full description
        if descriptions:
            full_description = " ".join(descriptions)
            return full_description
        else:
            return "The room contains various objects without clear spatial relationships."

    def simulate_physical_changes(self):
        """Simulate physical changes in the environment."""
        # Find all movable objects (like chairs)
        for obj_id, obj in self.objects.items():
            if obj.obj_type == "chair":
                # Randomly move chairs slightly - more frequent for demo
                if random.random() > 0.7:  # 30% chance to move
                    # Try to move the chair
                    max_attempts = 5  # Limit the number of attempts to find a valid position
                    for _ in range(max_attempts):
                        # Generate a potential new position
                        new_position = {
                            "x": obj.position["x"] + random.uniform(-0.15, 0.15),
                            "y": obj.position["y"] + random.uniform(-0.15, 0.15),
                            "z": obj.position["z"]
                        }

                        # Check if this position is valid (doesn't cause overlap)
                        if self.update_object_position(obj_id, new_position):
                            # Successfully moved, no overlap
                            break

    def simulate_step(self):
        """Simulate one step of the digital twin environment."""
        # Update each object's telemetry
        for obj_id, obj in self.objects.items():
            obj.simulate_telemetry_update()

        # Simulate workload spikes to create optimization opportunities
        self.simulate_workload_spikes()

        # Simulate physical changes
        self.simulate_physical_changes()

        # Update spatial relationships after physical changes
        self.update_spatial_relationships()

        # Periodically run resource allocation optimization (every 30 simulation steps)
        # Run more frequently if sim_time is low to demonstrate quickly

        if self.simulation_time % 30 == 0 and self.simulation_time > 0:
            # Create visualization of current resource state
            # pre_graph_path = self.create_resource_graph()

            # Run resource allocation optimization
            optimization_result = self.optimize_resource_allocation()

            # if optimization_result:
                # Create visualization after optimization
            post_graph_path = self.create_resource_graph()
                # print(f"Resource allocation before: {pre_graph_path}")
            if optimization_result:
                print(f"Resource allocation after optimization: {post_graph_path}")
            else:
                print(f"Resource allocation (no changes made): {post_graph_path}")

        # Save current state to history
        self.save_state()

        # Increment simulation time
        self.simulation_time += 1

    def save_state(self):
        """Save current state to history."""
        current_state = {
            "timestamp": time.time(),
            "simulation_time": self.simulation_time,
            "objects": {obj_id: obj.to_dict() for obj_id, obj in self.objects.items()},
            "scene_description": self.get_scene_description()  # Include scene description in state
        }
        self.history.append(current_state)

        # Limit history size to prevent memory issues
        if len(self.history) > 1000:
            self.history = self.history[-1000:]

    def start_simulation(self, interval=0.2):  # Faster updates for demo
        """Start continuous simulation in a separate thread."""
        if self.running:
            print("Simulation already running")
            return

        self.running = True
        print("Starting simulation thread...")

        def simulation_loop():
            print("Simulation loop started")
            while self.running:
                self.simulate_step()
                time.sleep(interval)
            print("Simulation loop ended")

        self.simulation_thread = threading.Thread(target=simulation_loop)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        print(f"Simulation thread started: {self.simulation_thread.is_alive()}")

    def stop_simulation(self):
        """Stop the simulation."""
        print("Stopping simulation...")
        self.running = False
        if self.simulation_thread:
            self.simulation_thread.join(timeout=1.0)
            print(f"Simulation thread stopped: {not self.simulation_thread.is_alive()}")

    def get_current_state(self):
        """Get the current state of all objects."""
        state = {obj_id: obj.to_dict() for obj_id, obj in self.objects.items()}
        state["scene_description"] = self.get_scene_description()  # Add scene description
        return state

    def simulate_workload_spikes(self):
        """
        Simulate periodic workload spikes to create opportunities for load balancing.
        This creates more realistic and varied load patterns across devices.
        """
        # Only create spikes occasionally (10% chance per step)
        if random.random() > 0.1:
            return

        # Find all computing devices
        computing_devices = [obj_id for obj_id, obj in self.objects.items()
                             if obj.obj_type in ['laptop', 'monitor']
                             and 'cpu_usage' in obj.telemetry]

        if not computing_devices:
            return

        # Choose a random device to spike
        device_id = random.choice(computing_devices)
        device = self.objects[device_id]

        # Define different spike patterns
        spike_patterns = [
            # (cpu_usage_change, temperature_change, duration_description)
            (30, 15, "heavy computation"),
            (50, 25, "system update"),
            (70, 35, "video rendering")
        ]

        # Choose a random spike pattern
        cpu_change, temp_change, description = random.choice(spike_patterns)

        # Apply the spike
        current_cpu = device.telemetry['cpu_usage']
        current_temp = device.telemetry['temperature']

        # Increase the values but keep within limits
        new_cpu = min(95, current_cpu + cpu_change)
        new_temp = min(90, current_temp + temp_change)

        # Update telemetry
        device.update_telemetry('cpu_usage', new_cpu)
        device.update_telemetry('temperature', new_temp)

        print(f"\nWorkload spike on {device_id}: {description}")
        print(f"  CPU: {current_cpu:.1f}% → {new_cpu:.1f}%")
        print(f"  Temperature: {current_temp:.1f}°C → {new_temp:.1f}°C")

    def optimize_resource_allocation(self, optimization_interval=10):
        """
        Optimize the allocation of computational resources across devices.
        """
        # Only run optimization at specified intervals
        if self.simulation_time % optimization_interval != 0:
            return False

        print(f"\nRunning resource allocation optimization at simulation time {self.simulation_time}...")

        # Identify computing devices in the digital twin
        computing_devices = [obj_id for obj_id, obj in self.objects.items()
                             if obj.obj_type in ['laptop', 'monitor']
                             and 'cpu_usage' in obj.telemetry]

        if len(computing_devices) < 2:
            print("Not enough computing devices for resource optimization")
            return False

        # Create a graph where nodes are computing devices
        G = nx.Graph()

        # Add nodes with current telemetry data
        for dev_id in computing_devices:
            dev = self.objects[dev_id]
            G.add_node(dev_id,
                       cpu_usage=dev.telemetry.get('cpu_usage', 0),
                       memory_usage=dev.telemetry.get('memory_usage', 0),
                       temperature=dev.telemetry.get('temperature', 40),
                       type=dev.obj_type,
                       max_capacity=dev.properties.get('cpu_cores', 4) * 25)  # Estimate max capacity

        # Add edges between all devices for potential workload sharing
        for i, dev1 in enumerate(computing_devices):
            for dev2 in computing_devices[i + 1:]:
                # Add edge with transfer capacity
                transfer_capacity = 25.0
                G.add_edge(dev1, dev2, capacity=transfer_capacity)

        # Store current system state for before/after comparison
        current_loads = {dev_id: G.nodes[dev_id]['cpu_usage'] for dev_id in computing_devices}
        current_temps = {dev_id: G.nodes[dev_id]['temperature'] for dev_id in computing_devices}
        max_capacities = {dev_id: G.nodes[dev_id]['max_capacity'] for dev_id in computing_devices}

        # Define target average load
        L_avg = sum(current_loads.values()) / len(current_loads)

        # Set optimization weights
        w_T = 0.4  # Weight for temperature
        w_L = 0.6  # Weight for load balance

        # Predictive models for temperature and power as functions of load
        def predict_temperature(device_id, load):
            """Predict temperature of a device based on CPU load."""
            base_temp = 35  # Base temperature when idle
            temp_factor = 0.5  # Temperature increase per % of CPU usage
            return base_temp + temp_factor * load

        # Define the objective function for optimization
        def objective_function(allocation_vector):
            """
            Objective function to minimize: Weighted sum of temperature and load imbalance.

            Returns:
                float: The weighted objective value to minimize
            """
            # Reshape allocation vector to per-device loads
            device_loads = dict(zip(computing_devices, allocation_vector))

            # Calculate predicted temperature and power for each device
            device_temps = {dev_id: predict_temperature(dev_id, load)
                            for dev_id, load in device_loads.items()}

            # Calculate the components of our objective function
            temp_component = sum(device_temps.values())
            load_balance_component = sum(abs(load - L_avg) for load in device_loads.values())

            # Weighted sum
            objective_value = (w_T * temp_component +
                               w_L * load_balance_component)

            return objective_value

        # Define constraints based on the graph
        constraints = []

        # 1. Capacity constraints for each device
        for dev_id in computing_devices:
            max_cap = max_capacities[dev_id]
            min_cap = 5.0  # Minimum load to keep devices responsive

            # Each device load must be within its capacity limits
            constraints.append({'type': 'ineq', 'fun': lambda x, dev=dev_id, idx=computing_devices.index(dev_id):
            max_cap - x[idx]})  # x[idx] <= max_cap
            constraints.append({'type': 'ineq', 'fun': lambda x, dev=dev_id, idx=computing_devices.index(dev_id):
            x[idx] - min_cap})  # x[idx] >= min_cap

        # 2. Total load conservation constraint
        # Sum of all loads must equal the sum of current loads
        total_current_load = sum(current_loads.values())
        constraints.append({'type': 'eq', 'fun': lambda x: sum(x) - total_current_load})

        # 3. Graph-based transfer constraints
        # For each edge, limit the amount of load that can be transferred
        for i, dev1 in enumerate(computing_devices):
            for j, dev2 in enumerate(computing_devices[i + 1:]):
                j = j + i + 1  # Adjust index
                if G.has_edge(dev1, dev2):
                    transfer_limit = G[dev1][dev2]['capacity']
                    # Limit the change in allocation between connected devices
                    constraints.append({'type': 'ineq', 'fun': lambda x, i=i, j=j:
                    transfer_limit - abs(x[i] - current_loads[computing_devices[i]] -
                                         (x[j] - current_loads[computing_devices[j]]))})

        # Initial guess: current loads
        initial_allocation = [current_loads[dev_id] for dev_id in computing_devices]

        # Run the optimization
        print("Starting optimization process...")
        result = minimize(
            objective_function,
            np.array(initial_allocation),
            method='SLSQP',  # Sequential Least Squares Programming
            constraints=constraints,
            options={'disp': True, 'maxiter': 100}
        )

        if result.success:
            optimized_allocation = result.x
            print(f"Optimization successful after {result.nit} iterations")

            # Check if the optimization made meaningful changes
            allocation_diff = np.array(optimized_allocation) - np.array(initial_allocation)
            if np.max(np.abs(allocation_diff)) < 3.0:
                print("Optimization resulted in minimal changes (<3% load difference)")

                # Even when no transfers needed, explain the current system state
                max_load_device = max(computing_devices, key=lambda x: current_loads[x])
                min_load_device = min(computing_devices, key=lambda x: current_loads[x])

                print(f"Current system state:")
                print(f"  Highest load: {max_load_device} at {current_loads[max_load_device]:.1f}%")
                print(f"  Lowest load: {min_load_device} at {current_loads[min_load_device]:.1f}%")
                print(f"  Load std deviation: {np.std(list(current_loads.values())):.1f}%")
                return False

            # Apply the optimized allocation
            transfers = []

            for i, dev_id in enumerate(computing_devices):
                new_load = optimized_allocation[i]
                load_change = new_load - current_loads[dev_id]

                if abs(load_change) > 0.5:  # Only process meaningful changes
                    # Find where this load comes from or goes to using the graph
                    if load_change > 0:  # This device receives load
                        # Find sources using the graph's neighbors
                        for neighbor in G.neighbors(dev_id):
                            if neighbor in computing_devices:
                                neighbor_idx = computing_devices.index(neighbor)
                                neighbor_change = initial_allocation[neighbor_idx] - optimized_allocation[neighbor_idx]

                                if neighbor_change > 0:  # This neighbor sends load
                                    transfer_amount = min(load_change, neighbor_change)
                                    if transfer_amount > 0.5:
                                        transfers.append({
                                            'from': neighbor,
                                            'to': dev_id,
                                            'amount': transfer_amount
                                        })
                                        load_change -= transfer_amount

            # Apply the transfers in the digital twin
            if transfers:
                print(f"Implementing {len(transfers)} workload transfers:")

                for transfer in transfers:
                    src_id = transfer['from']
                    dst_id = transfer['to']
                    amount = transfer['amount']

                    print(f"  Transfer {amount:.1f}% load from {src_id} to {dst_id}")

                    # Update telemetry to reflect the transfer
                    # Reduce source load
                    src_obj = self.objects[src_id]
                    new_src_load = current_loads[src_id] - amount
                    src_obj.update_telemetry('cpu_usage', new_src_load)

                    # Increase destination load
                    dst_obj = self.objects[dst_id]
                    new_dst_load = current_loads[dst_id] + amount
                    dst_obj.update_telemetry('cpu_usage', new_dst_load)

                    # Update temperatures based on load changes
                    new_src_temp = predict_temperature(src_id, new_src_load)
                    src_obj.update_telemetry('temperature', new_src_temp)

                    new_dst_temp = predict_temperature(dst_id, new_dst_load)
                    dst_obj.update_telemetry('temperature', new_dst_temp)

                    # Create a record of this optimization in spatial relations
                    src_obj.add_spatial_relation('shares_resources_with', dst_id)
                    if 'transfer_amounts' not in src_obj.metrics:
                        src_obj.metrics['transfer_amounts'] = {}
                    src_obj.metrics['transfer_amounts'][dst_id] = amount
                    dst_obj.add_spatial_relation('receives_resources_from', src_id)

                # Calculate improvement metrics
                new_loads = {dev_id: self.objects[dev_id].telemetry['cpu_usage'] for dev_id in computing_devices}
                new_temps = {dev_id: self.objects[dev_id].telemetry['temperature'] for dev_id in computing_devices}

                before_max_load = max(current_loads.values())
                after_max_load = max(new_loads.values())

                before_max_temp = max(current_temps.values())
                after_max_temp = max(new_temps.values())

                before_load_dev = np.std(list(current_loads.values()))
                after_load_dev = np.std(list(new_loads.values()))

                # Calculate the objective function value before and after
                before_obj = (w_T * sum(current_temps.values()) +
                              w_L * sum(abs(load - L_avg) for load in current_loads.values()))

                after_obj = (w_T * sum(new_temps.values()) +
                             w_L * sum(abs(load - L_avg) for load in new_loads.values()))

                improvement = (before_obj - after_obj) / before_obj * 100

                print(f"Optimization results:")
                print(f"  Objective function improvement: {improvement:.2f}%")
                print(f"  Max load: {before_max_load:.1f}% → {after_max_load:.1f}%")
                print(f"  Max temperature: {before_max_temp:.1f}°C → {after_max_temp:.1f}°C")
                print(f"  Load standard deviation: {before_load_dev:.1f}% → {after_load_dev:.1f}%")

                return True
            else:
                print("No workload transfers needed - system is already balanced")
                return False
        else:
            print(f"Optimization failed: {result.message}")
            return False

    # def optimize_resource_allocation(self, optimization_interval=10):
    #     """
    #     Transfer the allocation of computational resources across devices
    #     This is a rigid resource allocation, that is without optimization.
    #     """
    #     # Only run optimization at specified intervals
    #     if self.simulation_time % optimization_interval != 0:
    #         return False
    #
    #     print(f"\nRunning resource allocation optimization at simulation time {self.simulation_time}...")
    #
    #     # Identify computing devices in the digital twin
    #     computing_devices = [obj_id for obj_id, obj in self.objects.items()
    #                          if obj.obj_type in ['laptop', 'monitor']
    #                          and 'cpu_usage' in obj.telemetry]
    #
    #     if len(computing_devices) < 2:
    #         print("Not enough computing devices for resource optimization")
    #         return False
    #
    #     import networkx as nx
    #     import numpy as np
    #
    #     # Create a graph where nodes are computing devices
    #     G = nx.Graph()
    #
    #     # Add nodes with current telemetry data
    #     for dev_id in computing_devices:
    #         dev = self.objects[dev_id]
    #         G.add_node(dev_id,
    #                    cpu_usage=dev.telemetry.get('cpu_usage', 0),
    #                    memory_usage=dev.telemetry.get('memory_usage', 0),
    #                    temperature=dev.telemetry.get('temperature', 40),
    #                    type=dev.obj_type,
    #                    max_capacity=dev.properties.get('cpu_cores', 4) * 25)  # Estimate max capacity
    #
    #     # Add edges between all devices for potential workload sharing
    #     # In a real system, this would be based on network connectivity
    #     for i, dev1 in enumerate(computing_devices):
    #         for dev2 in computing_devices[i + 1:]:
    #             # Add edge with transfer capacity
    #             transfer_capacity = 25.0  # Increased capacity
    #             G.add_edge(dev1, dev2, capacity=transfer_capacity)
    #
    #     # Store current system state for before/after comparison
    #     current_loads = {dev_id: G.nodes[dev_id]['cpu_usage'] for dev_id in computing_devices}
    #     current_temps = {dev_id: G.nodes[dev_id]['temperature'] for dev_id in computing_devices}
    #     max_capacities = {dev_id: G.nodes[dev_id]['max_capacity'] for dev_id in computing_devices}
    #
    #     # Calculate load scores (combination of CPU and temperature)
    #     load_scores = {}
    #     for dev_id in computing_devices:
    #         cpu = current_loads[dev_id]
    #         temp = current_temps[dev_id]
    #         # Higher temperature contributes more to the load score
    #         load_scores[dev_id] = cpu + (temp - 35) * 0.7
    #
    #     # Find the average load score
    #     avg_load_score = sum(load_scores.values()) / len(load_scores)
    #     print(f"Average system load score: {avg_load_score:.1f}")
    #
    #     # Identify overloaded and underloaded devices
    #     # Use more sensitive thresholds: 5% difference instead of 10%
    #     overloaded = {}
    #     underloaded = {}
    #
    #     for dev_id in computing_devices:
    #         load_score = load_scores[dev_id]
    #         actual_load = current_loads[dev_id]
    #         temp = current_temps[dev_id]
    #
    #         # Lower threshold to detect more imbalances
    #         if load_score > avg_load_score + 5:  # Reduced from 10 to 5
    #             overloaded[dev_id] = {
    #                 'excess': load_score - avg_load_score,
    #                 'actual_load': actual_load,
    #                 'temperature': temp
    #             }
    #             print(f"  Overloaded: {dev_id} - Load: {actual_load:.1f}%, Temp: {temp:.1f}°C, Score: {load_score:.1f}")
    #         elif load_score < avg_load_score - 5:  # Reduced from 10 to 5
    #             # Calculate available capacity
    #             available_capacity = max_capacities[dev_id] - actual_load
    #             underloaded[dev_id] = {
    #                 'capacity': available_capacity,
    #                 'actual_load': actual_load,
    #                 'temperature': temp
    #             }
    #             print(
    #                 f"  Underloaded: {dev_id} - Load: {actual_load:.1f}%, Temp: {temp:.1f}°C, Score: {load_score:.1f}")
    #
    #     # Calculate load transfers to balance the system
    #     transfers = []
    #
    #     # Prioritize devices with highest temperature and load
    #     sorted_overloaded = sorted(overloaded.items(),
    #                                key=lambda x: (x[1]['temperature'], x[1]['actual_load']),
    #                                reverse=True)
    #
    #     for src_id, src_data in sorted_overloaded:
    #         remaining_excess = src_data['excess']
    #
    #         # Find all connected underloaded devices
    #         neighbors = list(G.neighbors(src_id))
    #         eligible_targets = [n for n in neighbors if n in underloaded]
    #
    #         # Sort targets by temperature (coolest first)
    #         eligible_targets.sort(key=lambda x: underloaded[x]['temperature'])
    #
    #         for dst_id in eligible_targets:
    #             if remaining_excess <= 0:
    #                 break
    #
    #             # Calculate how much load to transfer
    #             transfer_amount = min(
    #                 remaining_excess * 1.2,  # Convert from score to CPU percentage
    #                 underloaded[dst_id]['capacity'],
    #                 G[src_id][dst_id]['capacity']
    #             )
    #
    #             # Lower minimum transfer threshold to allow smaller transfers
    #             if transfer_amount > 3:  # Reduced from 5 to 3
    #                 transfers.append({
    #                     'from': src_id,
    #                     'to': dst_id,
    #                     'amount': transfer_amount
    #                 })
    #                 remaining_excess -= transfer_amount / 1.2  # Convert back to score
    #                 underloaded[dst_id]['capacity'] -= transfer_amount
    #
    #     # Apply the transfers in the digital twin
    #     if transfers:
    #         print(f"Implementing {len(transfers)} workload transfers:")
    #
    #         for transfer in transfers:
    #             src_id = transfer['from']
    #             dst_id = transfer['to']
    #             amount = transfer['amount']
    #
    #             print(f"  Transfer {amount:.1f}% load from {src_id} to {dst_id}")
    #
    #             # Update telemetry to reflect the transfer
    #             # Reduce source load
    #             src_obj = self.objects[src_id]
    #             new_src_load = max(5, src_obj.telemetry['cpu_usage'] - amount)
    #             src_obj.update_telemetry('cpu_usage', new_src_load)
    #
    #             # Increase destination load
    #             dst_obj = self.objects[dst_id]
    #             new_dst_load = min(95, dst_obj.telemetry['cpu_usage'] + amount)
    #             dst_obj.update_telemetry('cpu_usage', new_dst_load)
    #
    #             # Update temperatures based on load changes
    #             temp_factor = 0.3  # Temperature change per % of CPU (increased)
    #
    #             # Lower source temperature
    #             new_src_temp = max(35, src_obj.telemetry['temperature'] - (amount * temp_factor))
    #             src_obj.update_telemetry('temperature', new_src_temp)
    #
    #             # Increase destination temperature
    #             new_dst_temp = min(85, dst_obj.telemetry['temperature'] + (amount * temp_factor))
    #             dst_obj.update_telemetry('temperature', new_dst_temp)
    #
    #             # Create a record of this optimization in the spatial relations
    #             src_obj.add_spatial_relation('shares_resources_with', dst_id)
    #             # Store the transfer amount for visualization
    #             if 'transfer_amounts' not in src_obj.metrics:
    #                 src_obj.metrics['transfer_amounts'] = {}
    #             src_obj.metrics['transfer_amounts'][dst_id] = amount
    #             dst_obj.add_spatial_relation('receives_resources_from', src_id)
    #
    #         # Calculate improvement metrics
    #         before_max_load = max(current_loads.values())
    #         after_max_load = max(self.objects[dev_id].telemetry['cpu_usage'] for dev_id in computing_devices)
    #
    #         before_max_temp = max(current_temps.values())
    #         after_max_temp = max(self.objects[dev_id].telemetry['temperature'] for dev_id in computing_devices)
    #
    #         before_std_dev = np.std(list(current_loads.values()))
    #         after_std_dev = np.std([self.objects[dev_id].telemetry['cpu_usage'] for dev_id in computing_devices])
    #
    #         print(f"Optimization results:")
    #         print(f"  Max load: {before_max_load:.1f}% → {after_max_load:.1f}%")
    #         print(f"  Max temperature: {before_max_temp:.1f}°C → {after_max_temp:.1f}°C")
    #         print(f"  Load standard deviation: {before_std_dev:.1f}% → {after_std_dev:.1f}%")
    #
    #         return True
    #     else:
    #         print("No workload transfers needed - system is already balanced")
    #
    #         # Even when no transfers needed, explain the current system state
    #         max_load_device = max(computing_devices, key=lambda x: current_loads[x])
    #         min_load_device = min(computing_devices, key=lambda x: current_loads[x])
    #
    #         print(f"Current system state:")
    #         print(f"  Highest load: {max_load_device} at {current_loads[max_load_device]:.1f}%")
    #         print(f"  Lowest load: {min_load_device} at {current_loads[min_load_device]:.1f}%")
    #         print(f"  Load difference: {current_loads[max_load_device] - current_loads[min_load_device]:.1f}%")
    #         print(f"  Load standard deviation: {np.std(list(current_loads.values())):.1f}%")
    #
    #         return False

    def create_resource_graph(self):
        """Create a graph visualization of computing resource allocation."""
        import networkx as nx
        import matplotlib.pyplot as plt
        import numpy as np

        # Identify computing devices
        computing_devices = [obj_id for obj_id, obj in self.objects.items()
                             if obj.obj_type in ['laptop', 'monitor']
                             and 'cpu_usage' in obj.telemetry]

        if len(computing_devices) < 1:
            print("No computing devices for resource graph")
            return None

        # Create a directed graph for resource allocation
        G = nx.DiGraph()

        # Add computing devices as nodes
        for dev_id in computing_devices:
            dev = self.objects[dev_id]
            G.add_node(dev_id,
                       cpu_usage=dev.telemetry.get('cpu_usage', 0),
                       memory_usage=dev.telemetry.get('memory_usage', 0),
                       temperature=dev.telemetry.get('temperature', 40),
                       type=dev.obj_type)

        # Track transfer amounts for edge widths and labels
        transfer_amounts = {}

        # Add resource sharing edges
        for dev_id in computing_devices:
            dev = self.objects[dev_id]

            # Add edges for resource sharing relationships
            if 'shares_resources_with' in dev.spatial_relations:
                for target_id in dev.spatial_relations['shares_resources_with']:
                    if target_id in computing_devices:
                        # Get transfer amount if available (stored as a property in the relationship)
                        # Default to 10.0 if not found (for backward compatibility)
                        transfer_amount = 0

                        # First check if we stored the amount in the telemetry or metrics
                        if 'transfer_amounts' in dev.metrics and target_id in dev.metrics['transfer_amounts']:
                            transfer_amount = dev.metrics['transfer_amounts'][target_id]

                        # Add edge with transfer information
                        G.add_edge(dev_id, target_id,
                                   type='resource_sharing',
                                   amount=transfer_amount)

                        # Store for scaling edge widths
                        transfer_amounts[(dev_id, target_id)] = transfer_amount

        # Create visualization
        plt.figure(figsize=(10, 8))

        # Define node positions based on physical layout
        pos = {dev_id: (self.objects[dev_id].position["x"], self.objects[dev_id].position["y"])
               for dev_id in computing_devices}

        # Draw nodes with size proportional to CPU usage and color indicating temperature
        node_sizes = [300 + G.nodes[node]['cpu_usage'] * 5 for node in G.nodes()]

        # Create a colormap for temperature (blue=cool, red=hot)
        temps = [G.nodes[node]['temperature'] for node in G.nodes()]
        vmin = 35  # Minimum expected temperature
        vmax = 70  # Maximum expected temperature
        cmap = plt.cm.coolwarm

        # Draw nodes
        nodes = nx.draw_networkx_nodes(G, pos,
                                       node_size=node_sizes,
                                       node_color=temps,
                                       cmap=cmap,
                                       vmin=vmin, vmax=vmax)

        # Draw edges with width proportional to transfer amount
        # Get all edges and their attributes
        edges = [(u, v) for u, v, d in G.edges(data=True)]

        if edges:
            # Scale edge widths based on transfer amount
            edge_widths = [1 + G[u][v]['amount'] / 5 for u, v in edges]  # Width scaling

            # Draw the edges
            nx.draw_networkx_edges(G, pos,
                                   edgelist=edges,
                                   width=edge_widths,
                                   alpha=0.7,
                                   edge_color='green',
                                   arrowstyle='-|>',
                                   arrowsize=15)

            # Add edge labels with transfer amounts
            edge_labels = {(u, v): f"{G[u][v]['amount']:.1f}%" for u, v in edges}
            nx.draw_networkx_edge_labels(G, pos,
                                         edge_labels=edge_labels,
                                         font_size=9,
                                         font_color='darkgreen',
                                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        # Add node labels with CPU usage percentage
        labels = {node: f"{node}\n{G.nodes[node]['cpu_usage']:.0f}%" for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight='bold')

        # Add a colorbar for temperature
        plt.colorbar(nodes, label='Temperature (°C)')

        # Add title
        plt.title(f'Computing Resource Allocation (Time: {self.simulation_time}s)')

        # Add device workstations as background markers
        for dev_id in computing_devices:
            # Find desks related to this device
            for desk_id, desk in self.objects.items():
                if desk.obj_type != 'desk':
                    continue

                # Check if device is on this desk
                if ('on' in self.objects[dev_id].spatial_relations and
                        desk_id in self.objects[dev_id].spatial_relations['on']):
                    # Draw a rectangle representing the desk
                    desk_x = self.objects[desk_id].position["x"]
                    desk_y = self.objects[desk_id].position["y"]
                    desk_w = 1.0  # Approximate desk width
                    desk_h = 0.6  # Approximate desk height

                    plt.gca().add_patch(plt.Rectangle(
                        (desk_x - desk_w / 2, desk_y - desk_h / 2),
                        desk_w, desk_h,
                        fill=True, alpha=0.1, color='gray'))

        plt.axis('off')

        # Save the graph
        graph_path = f"dt_logs/resource_graph_{self.simulation_time}.png"
        os.makedirs(os.path.dirname(graph_path), exist_ok=True)
        plt.savefig(graph_path)
        plt.close()

        return graph_path


def run_dt_simulation_with_blender(blend_file_path="assets/scene.blend", num_frames=60, interval=0.5, optimize=True):
    """Run a digital twin simulation and visualize it with Blender."""
    # Create digital twin
    print("Creating digital twin...")
    dt = SimpleDTEnvironment()

    # Set room dimensions
    dt.set_room_dimensions(24, 24, 6)

    # Load the JSON file generated from the Blender scene
    with open("assets/scene_info.json", "r") as f:
        scene_info = json.load(f)

    def get_object_data(obj_name):
        """Find object data in the Blender scene JSON"""
        for name, data in scene_info["objects"].items():
            if obj_name.lower() in name.lower():
                return data
        return None

    def create_object_from_blender(obj_name, obj_type, additional_props=None):
        """Create a dt object with data from the Blender scene"""
        data = get_object_data(obj_name)
        if not data:
            print(f"Warning: {obj_name} not found in Blender data")
            return None

        # Create the object with location from Blender
        obj = dt.add_object(
            obj_name,
            obj_type,
            {"x": data["location"]["x"], "y": data["location"]["y"], "z": data["location"]["z"]},
            additional_props or {}
        )

        # Set real dimensions (dimensions × scale)
        obj.properties["dimensions"] = (
            data["dimensions"]["x"],
            data["dimensions"]["y"],
            data["dimensions"]["z"]
        )

        # Set rotation
        obj.properties["rotation_euler"] = (
            data["rotation_euler"]["x"],
            data["rotation_euler"]["y"],
            data["rotation_euler"]["z"]
        )

        return obj

    # Create objects using the helper function
    desk1 = create_object_from_blender("desk1", "desk")
    laptop1 = create_object_from_blender("laptop1", "laptop", {"cpu_cores": 4, "ram_gb": 16})
    mouse1 = create_object_from_blender("mouse1", "mouse")
    monitor1 = create_object_from_blender("monitor1", "monitor")
    keyboard1 = create_object_from_blender("keyboard1", "keyboard")

    desk2 = create_object_from_blender("desk2", "desk")
    chair2 = create_object_from_blender("chair2", "chair")

    desk3 = create_object_from_blender("desk3", "desk")
    chair3 = create_object_from_blender("chair3", "chair")
    mouse3 = create_object_from_blender("mouse2", "mouse")
    monitor3 = create_object_from_blender("monitor2", "monitor")
    keyboard3 = create_object_from_blender("keyboard2", "keyboard")

    # Update spatial relationships based on initial positions
    dt.update_spatial_relationships()

    # Display initial spatial relationships
    print("\nInitial Spatial Relationships:")
    for obj_id, obj in dt.objects.items():
        print(f"{obj_id} ({obj.obj_type}):")
        for relation_type, related_objects in obj.spatial_relations.items():
            if related_objects:
                related_str = ", ".join(related_objects)
                print(f"  {relation_type}: {related_str}")

    # Display initial scene description
    print("\nInitial Scene Description:")
    print(dt.get_scene_description())

    # Create logger
    logger = DTLogger(dt)

    # Create Blender visualizer
    visualizer = BlenderDTVisualizer(dt, logger, blend_file_path)

    # Initialize the visualizer (it will open the blend file)
    visualizer.initialize()

    # Start simulation
    dt.start_simulation(interval=0.2)

    try:
        print(f"\nRunning simulation for {num_frames} frames...")
        for frame in range(num_frames):
            # Let simulation run for a bit
            time.sleep(interval)

            # Log the current state
            state_file = logger.log_state(frame)

            # Create visualization frame in Blender
            img_path = visualizer.create_visualization_frame(frame)

            # Print progress
            if frame % 5 == 0 or frame == num_frames - 1:
                print(f"Frame {frame + 1}/{num_frames} completed - Time: {dt.simulation_time}s")
                print(f"  - State data: {os.path.basename(state_file)}")
                print(f"  - Image: {os.path.basename(img_path)}")

                # Print chair occupancy status
                chair_status = []
                for obj_id, obj in dt.objects.items():
                    if obj.obj_type == "chair" and "occupied" in obj.telemetry:
                        status = "occupied" if obj.telemetry["occupied"] else "empty"
                        chair_status.append(f"{obj_id}: {status}")

                if chair_status:
                    print(f"  - Chair status: {', '.join(chair_status)}")

                # Print updated scene description every 5 frames
                if frame % 10 == 0:
                    print(f"  - Scene: {dt.get_scene_description()}")
    finally:
        # Stop simulation and close logger
        dt.stop_simulation()
        logger.close()

        # Create compilation of all telemetry plots
        print("Creating final telemetry summary plots...")
        try:
            # Create a final summary plot of all telemetry
            final_telemetry_fig, axes = plt.subplots(3, 1, figsize=(10, 12), tight_layout=True)

            # Set titles
            axes[0].set_title('CPU Usage Over Time')
            axes[1].set_title('Memory Usage Over Time')
            axes[2].set_title('Temperature Over Time')

            # Plot data for all objects
            for obj_id, obj in dt.objects.items():
                if obj.obj_type in ["laptop", "monitor"]:
                    # Get telemetry history
                    time_data = [item[0] for item in obj.telemetry_history.get("cpu_usage", [])]

                    # Plot CPU usage
                    if "cpu_usage" in obj.telemetry_history and len(obj.telemetry_history["cpu_usage"]) > 1:
                        cpu_data = [item[1] for item in obj.telemetry_history["cpu_usage"]]
                        axes[0].plot(time_data, cpu_data, label=obj_id)

                    # Plot memory usage
                    if "memory_usage" in obj.telemetry_history and len(obj.telemetry_history["memory_usage"]) > 1:
                        memory_data = [item[1] for item in obj.telemetry_history["memory_usage"]]
                        axes[1].plot(time_data, memory_data, label=obj_id)

                    # Plot temperature
                    if "temperature" in obj.telemetry_history and len(obj.telemetry_history["temperature"]) > 1:
                        temp_data = [item[1] for item in obj.telemetry_history["temperature"]]
                        axes[2].plot(time_data, temp_data, label=obj_id)

            # Add formatting
            for ax in axes:
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.legend()
                ax.set_xlabel('Simulation Time (s)')

            axes[0].set_ylabel('CPU Usage (%)')
            axes[1].set_ylabel('Memory Usage (%)')
            axes[2].set_ylabel('Temperature (°C)')

            # Save the figure
            final_telemetry_path = os.path.join(logger.data_dir, "final_telemetry_summary.png")
            final_telemetry_fig.savefig(final_telemetry_path, dpi=100)
            plt.close(final_telemetry_fig)

            print(f"Final telemetry summary saved to: {final_telemetry_path}")
        except Exception as e:
            print(f"Error creating final telemetry summary: {e}")

        print("Simulation completed!")
        print(f"Log files saved to: {logger.run_dir}")
        print(f"  - Images: {logger.img_dir}")
        print(f"  - Data: {logger.data_dir}")


if __name__ == "__main__":
    # Get path to blend file from command line argument or use default
    blend_path = "assets/scene.blend"
    if len(sys.argv) > 1:
        blend_path = sys.argv[1]

    # Run simulation with Blender visualization
    run_dt_simulation_with_blender(
        blend_file_path=blend_path,
        num_frames=50,  # Reduce frame count for testing
        interval=1.0,
        optimize=True
    )
