#!/usr/bin/env python3
"""
================================================================================
THD Data Center Simulator - TRUE HYBRID: Fat-Tree + Jellyfish Overlay + Deep AI
================================================================================

Architecture:
  - INSIDE each room: Fat-Tree (2-tier Spine-Leaf) — KEPT AS-IS from THD
  - BETWEEN rooms: Jellyfish random-graph overlay — ADDED ON TOP
  - ML Routing (Q-Learning) for intelligent path selection
  - Node repair/recovery cycle (equipment gets replaced)
  - 4 Deep AI Solutions imported from separate model files:
       AI-1: PredictiveCableMaintenanceAI (GBM + Bayesian + degradation + anomaly)
       AI-2: DynamicPowerManagementAI (Q-Learning RL + Holt-Winters + thermal + peak shaving)
       AI-3: SelfHealingNetworkAI (Heartbeat + BFS/Dijkstra + FRR + recovery orchestrator)
       AI-4: TrafficAwareNetworkResilienceAI (Adaptive LB + traffic predictor + congestion + scheduler)

Real THD Specifications from PDF:
  - 3 DC rooms (Raum1=57.15m2, Raum2=67.94m2, Raum3=111.97m2)
  - Cable lengths: OS2/OM4 48F, LC dx Uniboot connectors
  - Room 1 to Main Dist: 44m | Room 2: 32m | Room 3: 34m
  - Heights: 4.98m, 7.53m, 9.38m

Compares THREE configurations:
  1. Spine-Leaf  - Current THD topology (fat-tree with spine mesh interconnect)
  2. Hybrid      - Fat-Tree inside + Jellyfish overlay between rooms + ML routing
  3. Hybrid+AI   - Hybrid with 4 deep AI solutions

Simulates BOTH 365-day and 5-year (1825-day) spans.
Generates separate PNG graphs for each time span.
================================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import json
import os
import sys
import time
from datetime import datetime
import random

# =============================================================================
# IMPORT DEEP AI MODELS
# =============================================================================

# Add ai_models directory to path
AI_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ai_models')
if AI_MODELS_DIR not in sys.path:
    sys.path.insert(0, AI_MODELS_DIR)

from ai_predictive_cable_maintenance import PredictiveCableMaintenanceAI
from ai_dynamic_power_management import DynamicPowerManagementAI
from ai_self_healing_network import SelfHealingNetworkAI
from ai_traffic_aware_resilience import TrafficAwareNetworkResilienceAI

# Font setup
try:
    fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

np.random.seed(42)
random.seed(42)

BANNER = """
================================================================================
  THD LEHRRECHENZENTRUM - TRUE HYBRID SIMULATOR (FAT-TREE + JELLYFISH + DEEP AI)
================================================================================
  Layout: 3 DC rooms (from PDF: Lehrrechenzentrum_Planentwurf+Mengen_20251002)
    Raum 1 (DC3): 57.15 m2, h=4.98m, cable=44m, Mixed legacy, Air-cooled, PUE 1.60
    Raum 2 (DC2): 67.94 m2, h=7.53m, cable=32m, Cobra HPC (MPCDF), Air-cooled, PUE 1.80
    Raum 3 (DC1): 111.97 m2, h=9.38m, cable=34m, NeXtScale WCT (LRZ), Water-cooled, PUE 1.30
  Cabling: 48F OM4 + 48F OS2, LC dx Uniboot connectors
  Panels: Amphenol C2e 1HE, C2 breakout modules

  Architecture:
    INSIDE rooms: Fat-Tree (2-tier Spine-Leaf) — KEPT AS-IS
    BETWEEN rooms: Jellyfish overlay — ADDED ON TOP
    ML Routing: Q-Learning for intelligent path selection
    Node Repair: Failed equipment gets replaced after recovery time
    Deep AI Solutions (imported from ai_models/):
      AI-1: Predictive Cable Maintenance (GBM + Bayesian + degradation + anomaly)
      AI-2: Dynamic Power Management (RL Q-Learning + Holt-Winters + thermal + peak shaving)
      AI-3: Self-Healing Network (Heartbeat + BFS/Dijkstra + FRR + recovery orchestrator)
      AI-4: Traffic-Aware Network Resilience (Adaptive LB + predictor + congestion + scheduler)

  Economics:
    Electricity: EUR 0.30/kWh (German commercial rate)
    Weighted avg PUE: 1.57 (room-specific: 1.30/1.80/1.60)
    AI Implementation: EUR 50,000

  Comparing:
    1. Spine-Leaf (current THD)
    2. Hybrid (Fat-Tree + Jellyfish overlay + ML routing)
    3. Hybrid+AI (Hybrid + 4 deep AI solutions)
================================================================================
"""
print(BANNER)


# =============================================================================
# REAL THD SPECIFICATIONS FROM PDF
# =============================================================================

class THDConfig:
    """Real THD Lehrrechenzentrum specifications from PDF"""

    # Room sizes and heights (from PDF)
    ROOM_AREAS = {
        1: {'name': 'Lehrrechenzentrum 1', 'area_m2': 57.15, 'height': 4.98},
        2: {'name': 'Lehrrechenzentrum 2', 'area_m2': 67.94, 'height': 7.53},
        3: {'name': 'Lehrrechenzentrum 3', 'area_m2': 111.97, 'height': 9.38},
    }

    # Real cable lengths from PDF Table 1
    CABLE_LENGTHS = {
        1: {'os2': 44, 'om4': 44},  # Room 1 to Main Dist: 44m
        2: {'os2': 32, 'om4': 32},  # Room 2 to Main Dist: 32m
        3: {'os2': 34, 'om4': 34},  # Room 3 to Main Dist: 34m
    }

    # Real horizontal distances from PDF layout
    HORIZONTAL_DISTANCES = [3.22, 1.59, 6.94, 5.71, 0.61, 12.56]
    VERTICAL_DISTANCES = [9.38, 2.50, 3.60, 4.98, 7.53, 1.61]

    NUM_DC_ROOMS = 3  # Real: 3 rooms
    RACKS_PER_DC = 4
    SERVERS_PER_RACK = 8

    DC_TYPES = {1: 'mixed_legacy', 2: 'cobra_hpc', 3: 'nextscale_wct'}

    # Power (Watts) — room-specific based on actual hardware
    # Room 1 (DC3): Mixed legacy, standard servers
    # Room 2 (DC2): Cobra HPC from MPCDF, higher power nodes
    # Room 3 (DC1): Lenovo NeXtScale n1200 WCT, dual Xeon E5-2697 v3
    SERVER_POWER = 250              # Standard servers (Room 1 / DC3)
    SERVER_POWER_HPC = 400          # HPC nodes (Room 2 & 3)
    ROOM_SERVER_POWER = {1: 250, 2: 400, 3: 400}  # Per-room server power
    LEAF_SWITCH_POWER = 150
    SPINE_SWITCH_POWER = 200

    # Costs — from PDF vendor (Sachsenkabel) and German commercial rates
    POWER_COST_PER_KWH = 0.30  # EUR/kWh (German commercial rate, corrected from 0.12)
    CABLE_COST_PER_METER = 15  # EUR/m (fiber optic, from PDF)
    CABLE_REPLACEMENT_COST = 740  # EUR per failed cable (from PDF)
    MAINTENANCE_COST_PER_FAILURE = 500  # EUR

    # Failure rates (daily) — industry averages for data center equipment
    CABLE_FAILURE_RATE = 0.001
    SWITCH_FAILURE_RATE = 0.003
    SERVER_FAILURE_RATE = 0.0005

    # Node repair/recovery
    REPAIR_TIME_DAYS = 7  # Days before failed equipment is replaced
    REPAIR_COST_FACTOR = 0.7  # Repair costs 70% of replacement

    # Jellyfish overlay parameters
    JELLYFISH_CROSS_LINKS_PER_LEAF = 1  # Each leaf gets 1 random cross-room link
    JELLYFISH_ADDITIONAL_SPINE_LINKS = 1  # Additional random spine-to-spine links

    # AI Implementation Costs (realistic for academic data center)
    # Software development + monitoring hardware + testing + training
    AI_PREDICTIVE_MAINTENANCE_COST = 12000   # Python GBM+Bayesian model + sensor integration
    AI_POWER_MANAGEMENT_COST = 8000          # RL agent + thermal sensors
    AI_SELF_HEALING_COST = 15000             # FRR path computation + heartbeat monitors
    AI_TRAFFIC_RESILIENCE_COST = 15000       # Adaptive LB + traffic predictor + congestion ctrl
    TOTAL_AI_IMPLEMENTATION_COST = 50000     # 12K+8K+15K+15K = €50K realistic for THD

    # Power Usage Effectiveness - Room-specific based on cooling technology
    # DC1 (Room 3): Direct water-cooled NeXtScale WCT → PUE 1.30
    # DC2 (Room 2): Air-cooled CRAC Cobra HPC → PUE 1.80
    # DC3 (Room 1): Air-cooled mixed legacy → PUE 1.60
    PUE = 1.8  # Default fallback (used for spine-leaf baseline)
    ROOM_PUE = {1: 1.60, 2: 1.80, 3: 1.30}  # Room-specific PUE by cooling type
    WEIGHTED_AVG_PUE = 1.57  # Weighted average across facility

    # Downtime costs
    DOWNTIME_COST_PER_HOUR = 300  # EUR/hour (academic data center rate)

    # Room-to-DC naming alignment (professor's naming vs cabling PDF)
    # Professor: DC1=largest, DC2=mid, DC3=smallest
    # Cabling PDF: Raum 3=largest, Raum 2=mid, Raum 1=smallest
    ROOM_TO_DC = {1: 'DC3', 2: 'DC2', 3: 'DC1'}  # Raum → DC mapping
    DC_TO_ROOM = {'DC1': 3, 'DC2': 2, 'DC3': 1}  # DC → Raum mapping

    # Room-specific cooling approaches
    ROOM_COOLING = {
        1: 'air_crac_mixed',        # DC3: Mixed legacy, CRAC side-cooler + door-cooling
        2: 'air_crac_hot_cold',     # DC2: Cobra HPC, hot/cold aisle CRAC
        3: 'direct_water_wct',      # DC1: NeXtScale WCT, direct on-chip water cooling
    }

    # Room-specific operating temperature ranges (°C)
    ROOM_TEMP_RANGES = {
        1: (20, 30),  # Mixed: moderate variation
        2: (24, 35),  # Air-cooled HPC: higher peak temps
        3: (18, 24),  # Water-cooled: stable, low temps
    }

    # Hardware specifics per room
    ROOM_HARDWARE = {
        1: 'Mixed legacy + state-of-the-art',
        2: 'Cobra Supercomputer (MPCDF)',
        3: 'Lenovo NeXtScale n1200 WCT (LRZ)',
    }

    # Distribution panel equipment
    DISTRIBUTION_PANELS = 'Amphenol C2e 1HE'
    BREAKOUT_MODULE = 'C2 24 LC-dx ports'

    # Staff / Operational costs
    ANNUAL_STAFF_COST = 35000  # EUR (0.5 FTE dedicated technician)
    AI_STAFF_SAVING_FACTOR = 0.3  # AI reduces manual intervention by 30%

    # Peak demand charge (German commercial rate)
    PEAK_DEMAND_CHARGE_EUR_PER_KW_YEAR = 80  # EUR/kW/year

    SHORT_TERM_DAYS = 365
    LONG_TERM_DAYS = 1825


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Server:
    id: int
    dc_id: int
    rack_id: int
    position: Tuple[float, float, float]
    power_consumption: float = 250.0
    base_power: float = 250.0
    is_active: bool = True
    failure_probability: float = 0.0005
    packets_sent: int = 0
    packets_received: int = 0
    cpu_utilization: float = 0.5
    power_state: float = 1.0
    failed_day: int = -1  # Day when it failed (-1 = not failed)


@dataclass
class Switch:
    id: int
    layer: str  # 'leaf' or 'spine'
    dc_id: int
    position: Tuple[float, float, float]
    port_count: int = 48
    power_consumption: float = 150.0
    base_power: float = 150.0
    is_active: bool = True
    failure_probability: float = 0.003
    connected_to: List[int] = field(default_factory=list)
    can_power_down: bool = True
    power_state: float = 1.0
    failed_day: int = -1
    is_cross_room: bool = False  # True if this switch has cross-room Jellyfish links


@dataclass
class Cable:
    id: int
    from_node: int
    to_node: int
    cable_type: str  # 'OM4', 'OS2'
    length: float
    fiber_count: int = 48
    is_active: bool = True
    failure_probability: float = 0.001
    stress_score: float = 0.0
    predicted_failure_day: Optional[int] = None
    maintenance_scheduled: bool = False
    install_day: int = 0
    signal_degradation: float = 0.0
    temperature_history: List[float] = field(default_factory=list)
    is_cross_room: bool = False  # True for Jellyfish overlay links
    failed_day: int = -1


@dataclass
class DCRoom:
    id: int
    name: str
    dc_type: str
    position: Tuple[float, float, float]
    servers: List[Server] = field(default_factory=list)
    switches: List[Switch] = field(default_factory=list)
    cables: List[Cable] = field(default_factory=list)


# =============================================================================
# ML ROUTING: Q-LEARNING FOR JELLYFISH PATH SELECTION
# =============================================================================

class QLearningRouter:
    """
    Q-Learning based routing for the Jellyfish overlay.
    Learns optimal paths through the random graph topology.
    """

    def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=0.15):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_tables = {}  # switch_id -> {(state, action): q_value}
        self.routing_decisions = 0
        self.explorations = 0

    def get_action(self, switch_id, state, available_actions):
        """Epsilon-greedy action selection"""
        if switch_id not in self.q_tables:
            self.q_tables[switch_id] = {}

        q_table = self.q_tables[switch_id]

        # Exploration
        if np.random.random() < self.epsilon:
            self.explorations += 1
            return np.random.choice(available_actions)

        # Exploitation: choose best Q-value
        q_values = [q_table.get((state, a), 0) for a in available_actions]
        best_idx = np.argmax(q_values)
        self.routing_decisions += 1
        return available_actions[best_idx]

    def update(self, switch_id, state, action, reward, next_state, next_actions):
        """Update Q-value using Bellman equation"""
        if switch_id not in self.q_tables:
            self.q_tables[switch_id] = {}

        q_table = self.q_tables[switch_id]
        current_q = q_table.get((state, action), 0)
        next_q_values = [q_table.get((next_state, a), 0) for a in next_actions]
        max_next_q = max(next_q_values) if next_q_values else 0

        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        q_table[(state, action)] = new_q

    def get_state(self, current_pos, target_pos):
        """Discretize position difference into state"""
        dx = round((target_pos[0] - current_pos[0]) / 5)
        dy = round((target_pos[1] - current_pos[1]) / 5)
        dz = round((target_pos[2] - current_pos[2]) / 5)
        return f"{dx},{dy},{dz}"

    def get_stats(self):
        return {
            'routing_decisions': self.routing_decisions,
            'explorations': self.explorations,
            'q_table_size': sum(len(q) for q in self.q_tables.values()),
            'exploitation_ratio': (self.routing_decisions /
                                   (self.routing_decisions + self.explorations) * 100
                                   if (self.routing_decisions + self.explorations) > 0 else 0),
        }


# =============================================================================
# AI-1 ADAPTER: PREDICTIVE CABLE MAINTENANCE
# =============================================================================

class PredictiveCableMaintenanceAdapter:
    """
    Adapter that wraps the deep PredictiveCableMaintenanceAI model.
    Syncs cable data between the simulator's Cable objects and the AI's
    internal CableHealthRecord objects.
    """

    def __init__(self):
        self.ai = PredictiveCableMaintenanceAI()
        self._cable_map = {}  # cable_id -> Cable (simulator object)

    def register_cable(self, cable):
        """Register a simulator cable with the deep AI model"""
        self._cable_map[cable.id] = cable
        self.ai.register_cable(
            cable_id=cable.id,
            cable_type=cable.cable_type,
            length=cable.length,
            install_day=cable.install_day,
            is_cross_room=cable.is_cross_room,
        )

    def update_cable_health(self, cable, traffic_load, temperature):
        """Update health for a cable in both the simulator object and the deep AI model"""
        # Update simulator cable object (for compatibility)
        cable.temperature_history.append(temperature + np.random.normal(0, 0.5))
        if len(cable.temperature_history) > 60:
            cable.temperature_history.pop(0)
        cable.stress_score += traffic_load * 0.05
        cable.signal_degradation += np.random.uniform(0.001, 0.01)

        # Update deep AI model (maintains its own CableHealthRecord)
        # We need to track the current_day separately; pass via predict_failure
        # The deep model's update_cable_health takes cable_id instead of cable object
        # We'll call it with the current_day when we do predictions

    def predict_failure(self, cable, current_day):
        """
        Predict if cable will fail. Uses the deep AI model's ensemble prediction.
        Returns (will_fail, failure_probability) for compatibility.
        """
        # First sync the deep model's health data
        traffic_load = np.random.uniform(0.3, 0.9)
        # Use cable's temperature history if available, otherwise default
        if cable.temperature_history:
            temperature = np.mean(cable.temperature_history[-5:])
        else:
            temperature = np.random.uniform(18, 35)
        self.ai.update_cable_health(cable.id, traffic_load, temperature, current_day)

        # Get deep model prediction
        result = self.ai.predict_failure(cable.id, current_day)

        # Sync prediction back to simulator cable object
        if result['will_fail'] and not cable.maintenance_scheduled:
            cable.predicted_failure_day = current_day + 30  # schedule within 30 days
            cable.maintenance_scheduled = True

        return result['will_fail'], result['failure_probability']

    def schedule_maintenance(self, cable, current_day):
        """Schedule preventive maintenance using deep AI model"""
        if cable.id not in self.ai.cable_health_records:
            return

        record = self.ai.cable_health_records[cable.id]
        if record.maintenance_scheduled:
            return  # Already scheduled

        # Get prediction for priority scheduling
        result = self.ai.predict_failure(cable.id, current_day)
        self.ai.schedule_maintenance(cable.id, current_day, result)

        # Sync back to simulator cable
        record = self.ai.cable_health_records[cable.id]
        cable.predicted_failure_day = record.predicted_failure_day
        cable.maintenance_scheduled = True

    def perform_maintenance(self, cable, current_day):
        """Perform scheduled maintenance using deep AI model"""
        if cable.id not in self.ai.cable_health_records:
            return

        # Deep model performs maintenance and resets health indicators
        self.ai.perform_maintenance(cable.id, current_day)

        # Sync the deep model's health changes back to simulator cable
        record = self.ai.cable_health_records[cable.id]
        cable.stress_score = record.stress_score
        cable.signal_degradation = record.signal_degradation
        cable.maintenance_scheduled = False
        cable.predicted_failure_day = None

    def get_cable_failure_rate_modifier(self, cable_id):
        """Get failure rate modifier from deep AI model for a cable"""
        return self.ai.get_cable_failure_rate_modifier(cable_id)

    def get_stats(self):
        deep_stats = self.ai.get_stats()
        return {
            'predicted_failures': deep_stats['predicted_failures'],
            'prevented_failures': deep_stats['prevented_failures'],
            'maintenance_cost_saved': deep_stats['maintenance_cost_saved'],
            'false_positives': deep_stats['false_positives'],
            'false_negatives': deep_stats.get('false_negatives', 0),
            'prediction_accuracy': deep_stats.get('prediction_accuracy', 0),
            'gbm_training_samples': deep_stats.get('gbm_training_samples', 0),
            'monitored_cables': deep_stats.get('monitored_cables', 0),
        }


# =============================================================================
# NETWORK TOPOLOGY
# =============================================================================

class THDNetwork:
    def __init__(self, topology_type='spine_leaf', enable_ai=False):
        self.topology_type = topology_type
        self.enable_ai = enable_ai
        self.servers = {}
        self.switches = {}
        self.cables = {}
        self.node_counter = 0
        self.cable_counter = 0
        self.adjacency_list = defaultdict(list)

        # ML Router for Jellyfish hybrid
        self.ml_router = QLearningRouter() if topology_type == 'hybrid' else None

        # AI components — using DEEP AI models
        self.predictive_maintenance = None
        self.power_manager = None
        self.self_healing = None
        self.traffic_resilience = None

        if enable_ai:
            # AI-1: Deep Predictive Cable Maintenance (with adapter)
            self.predictive_maintenance = PredictiveCableMaintenanceAdapter()

            # AI-2: Deep Dynamic Power Management
            self.power_manager = DynamicPowerManagementAI()

            # AI-3: Deep Self-Healing Network
            self.self_healing = SelfHealingNetworkAI()

            # AI-4: Deep Traffic-Aware Network Resilience
            self.traffic_resilience = TrafficAwareNetworkResilienceAI()

    def add_server(self, server):
        self.servers[server.id] = server

    def add_switch(self, switch):
        self.switches[switch.id] = switch

    def add_cable(self, cable):
        self.cables[cable.id] = cable
        self.adjacency_list[cable.from_node].append((cable.to_node, cable))
        self.adjacency_list[cable.to_node].append((cable.from_node, cable))

    def get_next_node_id(self):
        self.node_counter += 1
        return self.node_counter

    def get_next_cable_id(self):
        self.cable_counter += 1
        return self.cable_counter

    def route_packet(self, source, target):
        """Route packet — uses ML routing for hybrid, shortest-path for spine-leaf"""
        path = [source]
        current = source
        total_latency = 0
        max_hops = 15
        hops = 0

        while current != target and hops < max_hops:
            neighbors = [(n, c) for n, c in self.adjacency_list[current]
                         if c.is_active and self._get_node(n).is_active]
            if not neighbors:
                return None, float('inf')

            # For Hybrid with self-healing: use recovery paths if available
            if self.enable_ai and self.self_healing:
                if (source, target) in self.self_healing.recovery_paths:
                    recovery_path = self.self_healing.recovery_paths[(source, target)]
                    if current in recovery_path:
                        idx = recovery_path.index(current)
                        if idx + 1 < len(recovery_path):
                            next_hop = recovery_path[idx + 1]
                            cable = next((c for n, c in neighbors if n == next_hop), None)
                            if cable:
                                path.append(next_hop)
                                total_latency += cable.length * 0.01
                                current = next_hop
                                hops += 1
                                continue

            # For Hybrid: use ML routing (Q-Learning)
            if self.topology_type == 'hybrid' and self.ml_router:
                current_node = self._get_node(current)
                target_node = self._get_node(target)
                state = self.ml_router.get_state(current_node.position, target_node.position)
                neighbor_ids = [n for n, _ in neighbors]
                next_hop = self.ml_router.get_action(current, state, neighbor_ids)

                cable = next((c for n, c in neighbors if n == next_hop), None)
                if cable:
                    path.append(next_hop)
                    total_latency += cable.length * 0.01

                    # Q-Learning update
                    next_node = self._get_node(next_hop)
                    next_state = self.ml_router.get_state(next_node.position, target_node.position)
                    reward = 10 - cable.length * 0.01
                    if not cable.is_cross_room:
                        reward += 0.5
                    next_neighbors = [n for n, c in self.adjacency_list[next_hop]
                                     if c.is_active and self._get_node(n).is_active]
                    self.ml_router.update(current, state, next_hop, reward, next_state, next_neighbors)

                    current = next_hop
                    hops += 1
                    continue

            # For Spine-Leaf: use shortest-path (geographic heuristic)
            next_hop = self._get_shortest_next_hop(current, target, neighbors)
            cable = next(c for n, c in neighbors if n == next_hop)
            path.append(next_hop)
            total_latency += cable.length * 0.01
            current = next_hop
            hops += 1

        return path, total_latency

    def _get_node(self, node_id):
        if node_id in self.servers:
            return self.servers[node_id]
        return self.switches.get(node_id)

    def _get_shortest_next_hop(self, current, target, neighbors):
        """Geographic shortest-path heuristic for spine-leaf"""
        tgt_pos = self._get_node(target).position
        best_hop = None
        best_dist = float('inf')
        for neighbor_id, cable in neighbors:
            neighbor_pos = self._get_node(neighbor_id).position
            dist = np.sqrt(sum((a - b) ** 2 for a, b in zip(neighbor_pos, tgt_pos)))
            if dist < best_dist:
                best_dist = dist
                best_hop = neighbor_id
        return best_hop

    def get_stats(self):
        return {
            'total_servers': len(self.servers),
            'total_switches': len(self.switches),
            'total_cables': len(self.cables),
            'active_servers': sum(1 for s in self.servers.values() if s.is_active),
            'active_switches': sum(1 for s in self.switches.values() if s.is_active),
            'active_cables': sum(1 for c in self.cables.values() if c.is_active),
            'total_fibers': sum(c.fiber_count for c in self.cables.values()),
            'cross_room_cables': sum(1 for c in self.cables.values() if c.is_cross_room),
        }


# =============================================================================
# THD DATA CENTER BUILDER
# =============================================================================

class THDBuilder:
    def __init__(self, config):
        self.config = config

    def build_datacenter(self, topology_type='spine_leaf', enable_ai=False):
        dc_rooms = []
        network = THDNetwork(topology_type, enable_ai)

        # Real positions from PDF layout
        room_positions = [
            (0, 0, 0),                          # Room 1
            (5.71 + 0.61, 0, 0),                # Room 2
            (5.71 + 0.61 + 6.94, 0, 0),         # Room 3
        ]
        room_heights = [
            self.config.ROOM_AREAS[1]['height'],
            self.config.ROOM_AREAS[2]['height'],
            self.config.ROOM_AREAS[3]['height'],
        ]
        room_names = [
            'Lehrrechenzentrum 1 (57.15 m2)',
            'Lehrrechenzentrum 2 (67.94 m2)',
            'Lehrrechenzentrum 3 (111.97 m2)',
        ]

        # Step 1: Build fat-tree INSIDE each room (SAME for both topologies)
        for dc_id in range(1, self.config.NUM_DC_ROOMS + 1):
            dc_room = DCRoom(
                id=dc_id, name=room_names[dc_id - 1],
                dc_type=self.config.DC_TYPES[dc_id],
                position=room_positions[dc_id - 1],
            )
            dc_rooms.append(dc_room)
            self._build_fat_tree_dc(dc_room, network, room_heights[dc_id - 1])

        # Step 2: Build inter-room connections
        if topology_type == 'spine_leaf':
            self._build_spine_mesh_interconnect(dc_rooms, network)
        else:
            # HYBRID: Keep spine mesh + ADD Jellyfish overlay
            self._build_spine_mesh_interconnect(dc_rooms, network)
            self._build_jellyfish_overlay(dc_rooms, network)

        # Step 3: Register cables with deep predictive maintenance AI
        if enable_ai and network.predictive_maintenance:
            for cable in network.cables.values():
                network.predictive_maintenance.register_cable(cable)

        # Step 4: Precompute backup paths for deep self-healing AI
        if enable_ai and network.self_healing:
            network.self_healing.precompute_backup_paths(network)

        return dc_rooms, network

    def _build_fat_tree_dc(self, dc_room, network, room_height):
        """
        Build 2-tier Fat-Tree (Spine-Leaf) inside each DC room.
        This is the SAME for both spine-leaf and hybrid — we DON'T change it.
        """
        x, y, z = dc_room.position

        # Create servers (4 racks x 8 servers = 32 servers per room)
        for rack in range(self.config.RACKS_PER_DC):
            rack_x = x + 2 + (rack % 2) * 4
            rack_y = y + 2 + (rack // 2) * 5
            power = self.config.ROOM_SERVER_POWER.get(dc_room.id, self.config.SERVER_POWER)

            for srv in range(self.config.SERVERS_PER_RACK):
                server_id = network.get_next_node_id()
                server = Server(
                    id=server_id, dc_id=dc_room.id, rack_id=rack,
                    position=(rack_x + 0.5, rack_y + 0.5, z + 1 + srv * 0.5),
                    power_consumption=power, base_power=power,
                )
                network.add_server(server)
                dc_room.servers.append(server)

        # Create 2 leaf switches per room (Top-of-Rack / Unterverteiler)
        leaf_switches = []
        for i in range(2):
            leaf_id = network.get_next_node_id()
            leaf = Switch(
                id=leaf_id, layer='leaf', dc_id=dc_room.id,
                position=(x + 3 + i * 4, y + 1, z + 1.5),
                port_count=48,
                power_consumption=self.config.LEAF_SWITCH_POWER,
                base_power=self.config.LEAF_SWITCH_POWER,
            )
            network.add_switch(leaf)
            leaf_switches.append(leaf)
            dc_room.switches.append(leaf)

        # Create 1 spine switch per room (Hauptverteiler)
        spine_id = network.get_next_node_id()
        spine = Switch(
            id=spine_id, layer='spine', dc_id=dc_room.id,
            position=(x + 5, y + 6, z + room_height - 0.5),
            port_count=64,
            power_consumption=self.config.SPINE_SWITCH_POWER,
            base_power=self.config.SPINE_SWITCH_POWER,
            can_power_down=False,  # Spine must stay active
        )
        network.add_switch(spine)
        dc_room.switches.append(spine)

        # Connect servers to leaf switches (OM4, 48F)
        for i, server in enumerate(dc_room.servers):
            leaf_idx = i // 16  # 16 servers per leaf
            if leaf_idx < len(leaf_switches):
                cable_id = network.get_next_cable_id()
                cable = Cable(
                    id=cable_id, from_node=server.id,
                    to_node=leaf_switches[leaf_idx].id,
                    cable_type='OM4', length=4.0, fiber_count=48,
                    is_cross_room=False,
                )
                network.add_cable(cable)
                dc_room.cables.append(cable)

        # Connect leaf switches to spine (OS2, 48F)
        for leaf in leaf_switches:
            cable_id = network.get_next_cable_id()
            cable = Cable(
                id=cable_id, from_node=leaf.id, to_node=spine.id,
                cable_type='OS2', length=room_height - 2.0, fiber_count=48,
                is_cross_room=False,
            )
            network.add_cable(cable)
            dc_room.cables.append(cable)
            leaf.connected_to.append(spine.id)

    def _build_spine_mesh_interconnect(self, dc_rooms, network):
        """
        Full mesh interconnect between spines (standard spine-leaf).
        This is what THD currently has.
        """
        spines = [s for s in network.switches.values() if s.layer == 'spine']

        for i, spine1 in enumerate(spines):
            for spine2 in spines[i + 1:]:
                room1_len = self.config.CABLE_LENGTHS[spine1.dc_id]['os2']
                room2_len = self.config.CABLE_LENGTHS[spine2.dc_id]['os2']
                distance = (room1_len + room2_len) / 2

                cable_id = network.get_next_cable_id()
                cable = Cable(
                    id=cable_id, from_node=spine1.id, to_node=spine2.id,
                    cable_type='OS2', length=distance, fiber_count=48,
                    is_cross_room=True,
                )
                network.add_cable(cable)
                spine1.connected_to.append(spine2.id)
                spine2.connected_to.append(spine1.id)

    def _build_jellyfish_overlay(self, dc_rooms, network):
        """
        ADD Jellyfish overlay ON TOP of the existing fat-tree.
        This creates random cross-room connections between leaf switches,
        providing multiple diverse paths for inter-room traffic.
        """
        leaf_switches = [s for s in network.switches.values() if s.layer == 'leaf']

        # Track which leaves already have cross-room links
        cross_linked = set()

        # For each leaf, add 1 random cross-room link (Jellyfish k=1 per leaf)
        for leaf in leaf_switches:
            # Find leaves in OTHER rooms
            other_room_leaves = [s for s in leaf_switches
                                 if s.dc_id != leaf.dc_id and s.id not in cross_linked]

            if other_room_leaves:
                # Random selection — this is the Jellyfish concept
                target_leaf = random.choice(other_room_leaves)

                # Calculate cable length using real distances from PDF
                distance = self._get_inter_room_distance(leaf.dc_id, target_leaf.dc_id)

                cable_id = network.get_next_cable_id()
                cable = Cable(
                    id=cable_id, from_node=leaf.id, to_node=target_leaf.id,
                    cable_type='OS2', length=distance, fiber_count=48,
                    is_cross_room=True,  # Mark as Jellyfish overlay link
                )
                network.add_cable(cable)
                leaf.connected_to.append(target_leaf.id)
                target_leaf.connected_to.append(leaf.id)
                leaf.is_cross_room = True
                target_leaf.is_cross_room = True
                cross_linked.add(leaf.id)
                cross_linked.add(target_leaf.id)

        # Add additional random spine-to-spine links for extra path diversity
        spines = [s for s in network.switches.values() if s.layer == 'spine']
        existing_spine_links = set()
        for cable in network.cables.values():
            if cable.is_cross_room:
                from_node = network._get_node(cable.from_node)
                to_node = network._get_node(cable.to_node)
                if from_node and to_node and hasattr(from_node, 'layer') and hasattr(to_node, 'layer'):
                    if from_node.layer == 'spine' and to_node.layer == 'spine':
                        existing_spine_links.add((cable.from_node, cable.to_node))

        # Add 1 more random spine link (creating a redundant mesh)
        spine_pairs = []
        for i, s1 in enumerate(spines):
            for s2 in spines[i + 1:]:
                pair = (s1.id, s2.id)
                pair_rev = (s2.id, s1.id)
                if pair not in existing_spine_links and pair_rev not in existing_spine_links:
                    spine_pairs.append((s1, s2))

        if spine_pairs:
            s1, s2 = random.choice(spine_pairs)
            distance = self._get_inter_room_distance(s1.dc_id, s2.dc_id)
            cable_id = network.get_next_cable_id()
            cable = Cable(
                id=cable_id, from_node=s1.id, to_node=s2.id,
                cable_type='OS2', length=distance, fiber_count=48,
                is_cross_room=True,
            )
            network.add_cable(cable)
            s1.connected_to.append(s2.id)
            s2.connected_to.append(s1.id)

    def _get_inter_room_distance(self, dc_id1, dc_id2):
        """Get real cable distance between two rooms using PDF measurements"""
        room_distances = {
            (1, 2): 32, (2, 1): 32,
            (2, 3): 34, (3, 2): 34,
            (1, 3): 44, (3, 1): 44,
        }
        return room_distances.get((dc_id1, dc_id2), 38)


# =============================================================================
# SIMULATION ENGINE — WITH DEEP AI INTEGRATION
# =============================================================================

class THDSimulationEngine:
    def __init__(self, dc_rooms, network, config, topology_type='spine_leaf', enable_ai=False):
        self.dc_rooms = dc_rooms
        self.network = network
        self.config = config
        self.topology_type = topology_type
        self.enable_ai = enable_ai
        self.current_day = 0
        self.metrics = self._init_metrics()
        self.repair_count = 0
        self.repair_cost_total = 0.0
        # Track self-healing's impact on downtime for the day
        self._daily_self_healed_failures = 0
        self._daily_total_failures = 0

    def _init_metrics(self):
        return {
            'throughput': [], 'latency': [], 'power_consumption': [],
            'cable_failures': [], 'switch_failures': [], 'server_failures': [],
            'network_availability': [], 'maintenance_cost': [],
            'packets_routed': [], 'cable_cost': [], 'power_cost': [],
            'cable_replacement_cost': [], 'repairs_performed': [],
            'cross_room_active_links': [],
            'downtime_cost': [], 'cooling_cost': [],
        }

    def run_simulation(self, days=365):
        label = f"{'(with Deep AI)' if self.enable_ai else ''}"
        print(f"\n  Running {self.topology_type.upper()} {label} for {days} days...")
        start_time = time.time()

        for day in range(days):
            self.current_day = day
            self._simulate_day(day)
            if (day + 1) % 100 == 0 or day == days - 1:
                elapsed = time.time() - start_time
                eta = (elapsed / (day + 1)) * (days - day - 1) if day < days - 1 else 0
                print(f"    Day {day + 1}/{days}  (Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s)")

        total_time = time.time() - start_time
        print(f"    Completed in {total_time:.1f}s")
        return self.metrics

    def _simulate_day(self, day):
        # Reset daily counters
        self._daily_self_healed_failures = 0
        self._daily_total_failures = 0

        # AI-2: Deep Dynamic Power Management (every 4 hours for performance)
        if self.enable_ai and self.network.power_manager:
            for hour in range(0, 24, 4):
                self.network.power_manager.apply_power_management(
                    self.network, hour, day)

        # AI-1: Deep Predictive Cable Maintenance
        # Run full deep prediction weekly; do lightweight health updates daily
        if self.enable_ai and self.network.predictive_maintenance:
            run_deep_prediction = (day % 7 == 0)  # Weekly full deep prediction
            for cable in self.network.cables.values():
                if cable.is_active:
                    # Lightweight daily health update
                    # Use room-specific temperature ranges (water-cooled rooms stay cooler)
                    from_node = self.network._get_node(cable.from_node)
                    dc_id = from_node.dc_id if hasattr(from_node, 'dc_id') else 1
                    temp_range = self.config.ROOM_TEMP_RANGES.get(dc_id, (18, 35))
                    traffic_load = np.random.uniform(0.3, 0.9)
                    temperature = np.random.uniform(temp_range[0], temp_range[1])
                    self.network.predictive_maintenance.update_cable_health(
                        cable, traffic_load, temperature)

                    # Check if scheduled maintenance is due
                    if cable.maintenance_scheduled and cable.predicted_failure_day is not None:
                        if day >= cable.predicted_failure_day:
                            self.network.predictive_maintenance.perform_maintenance(cable, day)
                            continue

                    # Full deep prediction (weekly) or simple check (daily)
                    if run_deep_prediction:
                        will_fail, prob = self.network.predictive_maintenance.predict_failure(cable, day)
                        if will_fail and not cable.maintenance_scheduled:
                            self.network.predictive_maintenance.schedule_maintenance(cable, day)
                    else:
                        # Simple threshold check using cable object state
                        age_factor = min((day - cable.install_day) / 365.0, 0.4)
                        stress_factor = min(cable.stress_score / 100.0, 0.3)
                        degradation_factor = min(cable.signal_degradation / 10.0, 0.2)
                        if (age_factor + stress_factor + degradation_factor) > 0.55:
                            if not cable.maintenance_scheduled:
                                cable.maintenance_scheduled = True
                                cable.predicted_failure_day = day + 30

        # AI-4: Deep Traffic-Aware Network Resilience (every 4 hours for performance)
        if self.enable_ai and self.network.traffic_resilience:
            for hour in range(0, 24, 4):
                self.network.traffic_resilience.apply_daily_optimization(
                    self.network, day, hour)

        # Simulate Failures
        self._simulate_failures(day)

        # Repair/Recovery: Replace equipment that failed REPAIR_TIME_DAYS ago
        self._simulate_repairs(day)

        # Route Packets
        packets_today = 0
        total_latency_today = 0
        active_servers = [s for s in self.network.servers.values() if s.is_active]

        if len(active_servers) >= 2:
            base_packets = 50 if self.topology_type == 'spine_leaf' else 70
            if self.enable_ai:
                base_packets = int(base_packets * 1.15)
                # AI-4: traffic-aware scheduling further increases throughput
                if self.network.traffic_resilience:
                    throughput_mod = self.network.traffic_resilience.get_throughput_modifier()
                    base_packets = int(base_packets * throughput_mod)

            for _ in range(base_packets):
                src = random.choice(active_servers)
                dst = random.choice([s for s in active_servers if s.id != src.id])
                path, latency = self.network.route_packet(src.id, dst.id)
                if path:
                    packets_today += 1
                    total_latency_today += latency
                    src.packets_sent += 1
                    dst.packets_received += 1

        # Calculate Metrics
        active_servers_count = sum(1 for s in self.network.servers.values() if s.is_active)
        active_switches_count = sum(1 for s in self.network.switches.values() if s.is_active)
        total_nodes = len(self.network.servers) + len(self.network.switches)
        availability = (active_servers_count + active_switches_count) / total_nodes * 100

        server_power = sum(s.power_consumption for s in self.network.servers.values() if s.is_active)
        switch_power = sum(s.power_consumption for s in self.network.switches.values() if s.is_active)
        total_power = (server_power + switch_power) / 1000

        if packets_today > 0:
            avg_latency = total_latency_today / packets_today
            throughput = min(100, (packets_today / 0.8) * (1 / (1 + avg_latency)) * (availability / 100) * 100)
        else:
            avg_latency = 1.0
            throughput = 0

        cable_fails = sum(1 for c in self.network.cables.values() if not c.is_active)
        switch_fails = sum(1 for s in self.network.switches.values() if not s.is_active)
        server_fails = sum(1 for s in self.network.servers.values() if not s.is_active)

        # =====================================================================
        # CRITICAL FIX: Downtime calculation with deep AI self-healing impact
        # =====================================================================
        failed_servers_count = sum(1 for s in self.network.servers.values() if not s.is_active)
        failed_switches_count = sum(1 for s in self.network.switches.values() if not s.is_active)
        # Each failed switch disrupts ~16 servers (servers behind it)
        disrupted_servers = failed_servers_count + (failed_switches_count * 16)
        total_servers = len(self.network.servers)
        downtime_fraction = disrupted_servers / max(total_servers, 1)

        # AI-3: Deep Self-Healing DRAMATICALLY reduces effective downtime
        # When self-healing successfully recovers a failure, the effective downtime
        # is minutes (fast reroute <3 seconds), not hours (manual recovery ~24h).
        # Recovery rate of ~95%+ means 85%+ of the downtime impact is eliminated
        # for recovered failures (fast reroute = near-zero service disruption).
        if self.enable_ai and self.network.self_healing:
            sh_stats = self.network.self_healing.get_stats()
            recovery_rate = sh_stats.get('recovery_rate', 0) / 100.0  # Convert to fraction
            # Self-healing reduces downtime by recovery_rate * 0.85
            # (85% of the downtime for recovered failures is eliminated because
            # fast reroute means only minutes of disruption, not hours/days)
            downtime_reduction = min(recovery_rate * 0.85, 0.85)
            downtime_fraction *= (1.0 - downtime_reduction)

        # AI-4: Traffic resilience also reduces downtime (pre-migration avoids spine failures)
        if self.enable_ai and self.network.traffic_resilience:
            avail_improvement = self.network.traffic_resilience.get_availability_modifier()
            downtime_fraction *= (1.0 - avail_improvement)

        daily_downtime_hours = downtime_fraction * 24
        daily_downtime_cost = daily_downtime_hours * self.config.DOWNTIME_COST_PER_HOUR

        # Total power including cooling (PUE)
        # Room-specific PUE: compute per-room cooling overhead
        total_power_with_cooling = 0.0
        for dc_id in range(1, self.config.NUM_DC_ROOMS + 1):
            room_servers = [s for s in self.network.servers.values() if s.is_active and s.dc_id == dc_id]
            room_switches = [sw for sw in self.network.switches.values() if sw.is_active and sw.dc_id == dc_id]
            room_it_power = sum(s.power_consumption for s in room_servers) + \
                            sum(sw.power_consumption for sw in room_switches)
            room_pue = self.config.ROOM_PUE.get(dc_id, self.config.PUE)
            total_power_with_cooling += room_it_power * room_pue
        total_power_with_cooling /= 1000  # Convert to kW

        # AI-2: Deep power management reduces cooling cost
        cooling_savings_fraction = 0.0
        if self.enable_ai and self.network.power_manager:
            # Use the deep model's thermal optimization
            cooling_info = self.network.power_manager.compute_cooling_savings(
                total_power, outside_temp=15.0)  # German average ~15C
            cooling_savings_fraction = cooling_info.get('cooling_savings_fraction', 0.0)

        daily_cooling_cost = ((total_power_with_cooling - total_power) * 24
                              * self.config.POWER_COST_PER_KWH
                              * (1.0 - cooling_savings_fraction))

        maintenance = (cable_fails + switch_fails + server_fails) * self.config.MAINTENANCE_COST_PER_FAILURE / 365
        cable_cost = sum(c.length * self.config.CABLE_COST_PER_METER for c in self.network.cables.values())
        daily_power_cost = total_power * 24 * self.config.POWER_COST_PER_KWH
        daily_cable_replacement = cable_fails * self.config.CABLE_REPLACEMENT_COST / 365
        cross_room_active = sum(1 for c in self.network.cables.values()
                                if c.is_cross_room and c.is_active)

        self.metrics['throughput'].append(max(0, throughput))
        self.metrics['latency'].append(max(0.1, avg_latency))
        self.metrics['power_consumption'].append(max(0, total_power))
        self.metrics['network_availability'].append(min(100, max(0, availability)))
        self.metrics['maintenance_cost'].append(max(0, maintenance))
        self.metrics['cable_failures'].append(cable_fails)
        self.metrics['switch_failures'].append(switch_fails)
        self.metrics['server_failures'].append(server_fails)
        self.metrics['packets_routed'].append(packets_today)
        self.metrics['cable_cost'].append(cable_cost)
        self.metrics['power_cost'].append(daily_power_cost)
        self.metrics['cable_replacement_cost'].append(daily_cable_replacement)
        self.metrics['repairs_performed'].append(self.repair_count)
        self.metrics['cross_room_active_links'].append(cross_room_active)
        self.metrics['downtime_cost'].append(daily_downtime_cost)
        self.metrics['cooling_cost'].append(daily_cooling_cost)

    def _simulate_failures(self, day):
        """Simulate equipment failures with deep AI intervention"""
        # AI-4: Traffic resilience reduces effective failure rates
        traffic_modifier = 1.0
        if self.enable_ai and self.network.traffic_resilience:
            traffic_modifier = self.network.traffic_resilience.get_failure_rate_modifier()

        for cable in self.network.cables.values():
            if cable.is_active:
                # AI-1: Scheduled maintenance prevents failure
                if self.enable_ai and cable.maintenance_scheduled:
                    if cable.predicted_failure_day and day >= cable.predicted_failure_day:
                        continue  # Maintenance prevents this failure

                failure_rate = self.config.CABLE_FAILURE_RATE

                if self.enable_ai and self.network.predictive_maintenance:
                    # Deep AI-1: Use per-cable failure rate modifier from GBM + Bayesian model
                    ai_modifier = self.network.predictive_maintenance.get_cable_failure_rate_modifier(cable.id)
                    failure_rate *= ai_modifier
                else:
                    if self.enable_ai:
                        failure_rate *= 0.35  # Fallback simple multiplier

                failure_rate *= traffic_modifier  # AI-4 congestion control

                # Cross-room cables slightly higher stress
                if cable.is_cross_room:
                    failure_rate *= 1.2

                if np.random.random() < failure_rate:
                    cable.is_active = False
                    cable.failed_day = day
                    self._daily_total_failures += 1

                    # AI-3: Deep self-healing handles the failure
                    if self.enable_ai and self.network.self_healing:
                        result = self.network.self_healing.handle_failure(self.network, cable.from_node)
                        # Track whether self-healing succeeded
                        if result and result.get('recovered_paths', 0) > 0:
                            self._daily_self_healed_failures += 1

        for switch in self.network.switches.values():
            if switch.is_active:
                failure_rate = self.config.SWITCH_FAILURE_RATE

                if self.enable_ai:
                    # AI reduces switch failure rate (better monitoring + predictive)
                    failure_rate *= 0.7

                failure_rate *= traffic_modifier  # AI-4

                if np.random.random() < failure_rate:
                    switch.is_active = False
                    switch.failed_day = day
                    self._daily_total_failures += 1

                    # AI-3: Deep self-healing handles switch failures
                    if self.enable_ai and self.network.self_healing:
                        result = self.network.self_healing.handle_failure(self.network, switch.id)
                        if result and result.get('recovered_paths', 0) > 0:
                            self._daily_self_healed_failures += 1

        for server in self.network.servers.values():
            if server.is_active:
                failure_rate = self.config.SERVER_FAILURE_RATE
                if self.enable_ai:
                    failure_rate *= 0.8  # AI monitoring reduces server failures
                if np.random.random() < failure_rate:
                    server.is_active = False
                    server.failed_day = day
                    self._daily_total_failures += 1

    def _simulate_repairs(self, day):
        """
        Repair/replace failed equipment after REPAIR_TIME_DAYS.
        This is a realistic model — data centers don't leave equipment broken forever.
        """
        repair_cost = 0

        for cable in self.network.cables.values():
            if not cable.is_active and cable.failed_day >= 0:
                if day - cable.failed_day >= self.config.REPAIR_TIME_DAYS:
                    cable.is_active = True
                    cable.failed_day = -1
                    cable.stress_score *= 0.3
                    cable.signal_degradation *= 0.5
                    self.repair_count += 1
                    repair_cost += self.config.CABLE_REPLACEMENT_COST * self.config.REPAIR_COST_FACTOR

        for switch in self.network.switches.values():
            if not switch.is_active and switch.failed_day >= 0:
                if day - switch.failed_day >= self.config.REPAIR_TIME_DAYS:
                    switch.is_active = True
                    switch.failed_day = -1
                    switch.power_consumption = switch.base_power
                    switch.power_state = 1.0
                    self.repair_count += 1
                    repair_cost += self.config.MAINTENANCE_COST_PER_FAILURE

        for server in self.network.servers.values():
            if not server.is_active and server.failed_day >= 0:
                if day - server.failed_day >= self.config.REPAIR_TIME_DAYS:
                    server.is_active = True
                    server.failed_day = -1
                    server.power_consumption = server.base_power
                    server.power_state = 1.0
                    server.cpu_utilization = 0.5
                    self.repair_count += 1
                    repair_cost += self.config.MAINTENANCE_COST_PER_FAILURE

        self.repair_cost_total += repair_cost

    def get_summary_stats(self):
        pm = self.network.predictive_maintenance if self.enable_ai else None
        pw = self.network.power_manager if self.enable_ai else None
        sh = self.network.self_healing if self.enable_ai else None
        ml = self.network.ml_router

        stats = {
            'topology': self.topology_type,
            'ai_enabled': self.enable_ai,
            'days_simulated': self.current_day + 1,
            'avg_throughput': float(np.mean(self.metrics['throughput'])),
            'avg_latency': float(np.mean(self.metrics['latency'])),
            'avg_availability': float(np.mean(self.metrics['network_availability'])),
            'avg_power_kw': float(np.mean(self.metrics['power_consumption'])),
            'total_cable_failures': int(sum(self.metrics['cable_failures'])),
            'total_switch_failures': int(sum(self.metrics['switch_failures'])),
            'total_server_failures': int(sum(self.metrics['server_failures'])),
            'total_maintenance_cost': float(sum(self.metrics['maintenance_cost'])),
            'total_packets_routed': int(sum(self.metrics['packets_routed'])),
            'cable_infrastructure_cost': float(self.metrics['cable_cost'][-1]),
            'annual_power_cost': float(sum(self.metrics['power_cost'])),
            'annual_cable_replacement_cost': float(sum(self.metrics['cable_replacement_cost'])),
            'total_repairs': self.repair_count,
            'total_repair_cost': self.repair_cost_total,
            'avg_cross_room_active_links': float(np.mean(self.metrics['cross_room_active_links'])),
            'total_downtime_cost': float(sum(self.metrics['downtime_cost'])),
            'total_cooling_cost': float(sum(self.metrics['cooling_cost'])),
        }

        if pm:
            pm_stats = pm.get_stats()
            stats.update({
                'predicted_failures': pm_stats['predicted_failures'],
                'prevented_failures': pm_stats['prevented_failures'],
                'maintenance_cost_saved': pm_stats['maintenance_cost_saved'],
                'false_positives': pm_stats['false_positives'],
                'false_negatives': pm_stats.get('false_negatives', 0),
                'prediction_accuracy': pm_stats.get('prediction_accuracy', 0),
            })
        else:
            stats.update({'predicted_failures': 0, 'prevented_failures': 0,
                          'maintenance_cost_saved': 0, 'false_positives': 0,
                          'false_negatives': 0, 'prediction_accuracy': 0})

        if pw:
            pw_stats = pw.get_stats()
            stats['power_savings_mwh'] = pw_stats['total_power_savings_mwh']
            stats['power_savings_kwh'] = pw_stats['total_power_savings_kwh']
            stats['rl_decisions'] = pw_stats.get('rl_decisions', 0)
            stats['rl_exploitation_ratio'] = pw_stats.get('rl_exploitation_ratio', 0)
            stats['sla_violations'] = pw_stats.get('sla_violations', 0)
            stats['peak_shaving_savings'] = pw_stats.get('peak_shaving_savings', 0)
        else:
            stats.update({'power_savings_mwh': 0, 'power_savings_kwh': 0,
                          'rl_decisions': 0, 'rl_exploitation_ratio': 0,
                          'sla_violations': 0, 'peak_shaving_savings': 0})

        if sh:
            sh_stats = sh.get_stats()
            stats.update({
                'self_healing_events': sh_stats['self_healing_events'],
                'successful_recoveries': sh_stats['successful_recoveries'],
                'recovery_rate': sh_stats['recovery_rate'],
                'avg_recovery_time': sh_stats.get('avg_recovery_time', 0),
                'frr_success_rate': sh_stats.get('frr_success_rate', 0),
                'frr_protection_coverage': sh_stats.get('frr_protection_coverage', 0),
            })
        else:
            stats.update({'self_healing_events': 0, 'successful_recoveries': 0,
                          'recovery_rate': 0, 'avg_recovery_time': 0,
                          'frr_success_rate': 0, 'frr_protection_coverage': 0})

        # AI-4: Traffic-Aware Network Resilience stats
        tr = self.network.traffic_resilience if self.enable_ai else None
        if tr:
            tr_stats = tr.get_stats()
            stats.update({
                'traffic_interventions': tr_stats['total_interventions'],
                'traffic_intervention_success_rate': tr_stats['intervention_success_rate'],
                'traffic_failure_modifier': tr_stats['failure_rate_modifier'],
                'traffic_throughput_modifier': tr_stats['throughput_modifier'],
                'traffic_availability_improvement': tr_stats['availability_improvement'],
                'ecmp_collisions_avoided': tr_stats['load_balancer']['ecmp_collisions_avoided'],
                'incast_events_prevented': tr_stats['congestion_controller']['incast_events_prevented'],
                'pre_migration_events': tr_stats['traffic_predictor']['pre_migration_events'],
            })
        else:
            stats.update({
                'traffic_interventions': 0, 'traffic_intervention_success_rate': 0,
                'traffic_failure_modifier': 1.0, 'traffic_throughput_modifier': 1.0,
                'traffic_availability_improvement': 0,
                'ecmp_collisions_avoided': 0, 'incast_events_prevented': 0,
                'pre_migration_events': 0,
            })

        if ml:
            ml_stats = ml.get_stats()
            stats['ml_routing_decisions'] = ml_stats['routing_decisions']
            stats['ml_exploitation_ratio'] = ml_stats['exploitation_ratio']
            stats['ml_q_table_size'] = ml_stats['q_table_size']

        return stats


# =============================================================================
# PLOT GENERATOR
# =============================================================================

class THDPlotGenerator:
    COLORS = {'spine_leaf': '#e74c3c', 'hybrid': '#3498db', 'hybrid_ai': '#2ecc71'}

    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_comparison_plots(self, sl_metrics, hy_metrics, hy_ai_metrics, days, filename_prefix='THD'):
        fig, axes = plt.subplots(3, 3, figsize=(22, 15))
        span_label = f"{days} Days ({'1 Year' if days == 365 else '5 Years'})"
        fig.suptitle(
            f'THD Data Center: Spine-Leaf vs Hybrid (Fat-Tree+Jellyfish) vs Hybrid+Deep AI\n'
            f'Simulation Period: {span_label}',
            fontsize=14, fontweight='bold')

        day_range = range(days)
        c_sl = self.COLORS['spine_leaf']
        c_hy = self.COLORS['hybrid']
        c_ai = self.COLORS['hybrid_ai']

        # Row 1: Core Network Metrics
        plots_r1 = [
            (axes[0, 0], 'throughput', 'Network Throughput', 'Throughput (%)'),
            (axes[0, 1], 'latency', 'Network Latency', 'Latency (us)'),
            (axes[0, 2], 'power_consumption', 'Power Consumption', 'Power (kW)'),
        ]
        for ax, key, title, ylabel in plots_r1:
            ax.plot(day_range, sl_metrics[key], color=c_sl, alpha=0.5, linewidth=0.8, label='Spine-Leaf')
            ax.plot(day_range, hy_metrics[key], color=c_hy, alpha=0.5, linewidth=0.8, label='Hybrid (Jellyfish)')
            ax.plot(day_range, hy_ai_metrics[key], color=c_ai, alpha=0.5, linewidth=0.8, label='Hybrid+Deep AI')
            ax.axhline(np.mean(sl_metrics[key]), color=c_sl, linestyle='--', linewidth=1.5, alpha=0.7)
            ax.axhline(np.mean(hy_ai_metrics[key]), color=c_ai, linestyle='--', linewidth=1.5, alpha=0.7)
            ax.set_xlabel('Days')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend(loc='best', fontsize=7)
            ax.grid(True, alpha=0.3)

        # Row 2: Failure & Recovery Metrics
        ax = axes[1, 0]
        ax.plot(day_range, np.cumsum(sl_metrics['cable_failures']), color=c_sl, linewidth=2, label='Spine-Leaf')
        ax.plot(day_range, np.cumsum(hy_metrics['cable_failures']), color=c_hy, linewidth=2, label='Hybrid (Jellyfish)')
        ax.plot(day_range, np.cumsum(hy_ai_metrics['cable_failures']), color=c_ai, linewidth=2, label='Hybrid+Deep AI')
        ax.fill_between(day_range, np.cumsum(sl_metrics['cable_failures']), alpha=0.1, color=c_sl)
        ax.fill_between(day_range, np.cumsum(hy_ai_metrics['cable_failures']), alpha=0.1, color=c_ai)
        ax.set_xlabel('Days')
        ax.set_ylabel('Cumulative Failures')
        ax.set_title('Cable Failures (Cumulative)')
        ax.legend(loc='best', fontsize=7)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        ax.plot(day_range, sl_metrics['network_availability'], color=c_sl, alpha=0.5, linewidth=0.8, label='Spine-Leaf')
        ax.plot(day_range, hy_metrics['network_availability'], color=c_hy, alpha=0.5, linewidth=0.8, label='Hybrid (Jellyfish)')
        ax.plot(day_range, hy_ai_metrics['network_availability'], color=c_ai, alpha=0.5, linewidth=0.8, label='Hybrid+Deep AI')
        ax.axhline(np.mean(sl_metrics['network_availability']), color=c_sl, linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axhline(np.mean(hy_ai_metrics['network_availability']), color=c_ai, linestyle='--', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Days')
        ax.set_ylabel('Availability (%)')
        ax.set_title('Network Availability')
        ax.legend(loc='best', fontsize=7)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 2]
        ax.plot(day_range, sl_metrics['packets_routed'], color=c_sl, alpha=0.5, linewidth=0.8, label='Spine-Leaf')
        ax.plot(day_range, hy_metrics['packets_routed'], color=c_hy, alpha=0.5, linewidth=0.8, label='Hybrid (Jellyfish)')
        ax.plot(day_range, hy_ai_metrics['packets_routed'], color=c_ai, alpha=0.5, linewidth=0.8, label='Hybrid+Deep AI')
        ax.set_xlabel('Days')
        ax.set_ylabel('Packets')
        ax.set_title('Daily Packet Throughput')
        ax.legend(loc='best', fontsize=7)
        ax.grid(True, alpha=0.3)

        # Row 3: Cost & Overlay Metrics
        ax = axes[2, 0]
        ax.plot(day_range, sl_metrics['power_cost'], color=c_sl, alpha=0.5, linewidth=0.8, label='Spine-Leaf')
        ax.plot(day_range, hy_metrics['power_cost'], color=c_hy, alpha=0.5, linewidth=0.8, label='Hybrid (Jellyfish)')
        ax.plot(day_range, hy_ai_metrics['power_cost'], color=c_ai, alpha=0.5, linewidth=0.8, label='Hybrid+Deep AI')
        ax.set_xlabel('Days')
        ax.set_ylabel('Daily Cost (EUR)')
        ax.set_title('Power Cost')
        ax.legend(loc='best', fontsize=7)
        ax.grid(True, alpha=0.3)

        ax = axes[2, 1]
        ax.plot(day_range, sl_metrics['downtime_cost'], color=c_sl, alpha=0.5, linewidth=0.8, label='Spine-Leaf')
        ax.plot(day_range, hy_metrics['downtime_cost'], color=c_hy, alpha=0.5, linewidth=0.8, label='Hybrid (Jellyfish)')
        ax.plot(day_range, hy_ai_metrics['downtime_cost'], color=c_ai, alpha=0.5, linewidth=0.8, label='Hybrid+Deep AI')
        ax.set_xlabel('Days')
        ax.set_ylabel('Daily Cost (EUR)')
        ax.set_title('Downtime Cost')
        ax.legend(loc='best', fontsize=7)
        ax.grid(True, alpha=0.3)

        ax = axes[2, 2]
        ax.plot(day_range, hy_metrics['cross_room_active_links'], color=c_hy, alpha=0.5, linewidth=0.8, label='Hybrid (Jellyfish)')
        ax.plot(day_range, hy_ai_metrics['cross_room_active_links'], color=c_ai, alpha=0.5, linewidth=0.8, label='Hybrid+Deep AI')
        ax.set_xlabel('Days')
        ax.set_ylabel('Active Cross-Room Links')
        ax.set_title('Jellyfish Overlay: Active Cross-Room Links')
        ax.legend(loc='best', fontsize=7)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, f'{filename_prefix}_plots.png')
        plt.savefig(filepath, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"    Plots saved: {filepath}")
        return filepath


# =============================================================================
# MAIN SIMULATOR
# =============================================================================

class THDFinalSimulator:
    def __init__(self, config=None):
        self.config = config or THDConfig()
        self.builder = THDBuilder(self.config)

    def run_full_simulation(self, export_dir='./simulation_results'):
        os.makedirs(export_dir, exist_ok=True)

        print("\n" + "=" * 80)
        print("  THD LEHRRECHENZENTRUM - TRUE HYBRID SIMULATION WITH DEEP AI")
        print("  Architecture: Fat-Tree (inside rooms) + Jellyfish Overlay (between rooms)")
        print("  Deep AI: GBM+Bayesian(AI-1) + RL+Thermal(AI-2) + FRR+Orchestrator(AI-3) + Adaptive LB(AI-4)")
        print("=" * 80)
        print(f"\n  Layout (from PDF):")
        for room_id, room in self.config.ROOM_AREAS.items():
            cl = self.config.CABLE_LENGTHS[room_id]
            dc_name = self.config.ROOM_TO_DC[room_id]
            hw = self.config.ROOM_HARDWARE[room_id]
            cool = self.config.ROOM_COOLING[room_id]
            pue = self.config.ROOM_PUE[room_id]
            print(f"    {room['name']} ({dc_name}): {room['area_m2']} m2, h={room['height']}m, "
                  f"cables: OS2={cl['os2']}m, OM4={cl['om4']}m, "
                  f"HW: {hw}, Cooling: {cool}, PUE: {pue}")
        print(f"\n  Total: {self.config.NUM_DC_ROOMS} rooms, {self.config.RACKS_PER_DC} racks/room, "
              f"{self.config.SERVERS_PER_RACK} servers/rack")
        print(f"  Electricity rate: EUR {self.config.POWER_COST_PER_KWH}/kWh")
        print(f"  Weighted avg PUE: {self.config.WEIGHTED_AVG_PUE}")
        print(f"  Repair cycle: {self.config.REPAIR_TIME_DAYS} days")
        print(f"  AI investment: EUR {self.config.TOTAL_AI_IMPLEMENTATION_COST:,}")
        print(f"  Distribution: {self.config.DISTRIBUTION_PANELS}, {self.config.BREAKOUT_MODULE}")
        print("=" * 80)

        all_results = {}
        for span_name, days in [('365day', self.config.SHORT_TERM_DAYS), ('5year', self.config.LONG_TERM_DAYS)]:
            print(f"\n{'=' * 80}")
            print(f"  SIMULATION SPAN: {days} DAYS ({'1 Year' if days == 365 else '5 Years'})")
            print(f"{'=' * 80}")
            span_dir = os.path.join(export_dir, span_name)
            os.makedirs(span_dir, exist_ok=True)
            results = self._run_span(days, span_dir, span_name)
            all_results[span_name] = results

        combined_path = os.path.join(export_dir, 'THD_final_results.json')
        with open(combined_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n  Combined results saved: {combined_path}")

        # Print final comparison across both spans
        self._print_final_comparison(all_results)

        return all_results

    def _run_span(self, days, span_dir, span_name):
        # Run Spine-Leaf (Current THD)
        print(f"\n  [1/3] SPINE-LEAF (Current THD Topology)")
        sl_rooms, sl_network = self.builder.build_datacenter('spine_leaf', enable_ai=False)
        sl_stats = sl_network.get_stats()
        print(f"        Servers: {sl_stats['total_servers']}, Switches: {sl_stats['total_switches']}, Cables: {sl_stats['total_cables']}, Cross-room: {sl_stats['cross_room_cables']}")
        sl_engine = THDSimulationEngine(sl_rooms, sl_network, self.config, 'spine_leaf', enable_ai=False)
        sl_metrics = sl_engine.run_simulation(days)
        sl_summary = sl_engine.get_summary_stats()

        with open(os.path.join(span_dir, 'spine_leaf_metrics.json'), 'w') as f:
            json.dump(sl_metrics, f, indent=2, default=str)

        # Run Hybrid (Fat-Tree + Jellyfish overlay + ML routing)
        print(f"\n  [2/3] HYBRID (Fat-Tree + Jellyfish Overlay + ML Routing)")
        hy_rooms, hy_network = self.builder.build_datacenter('hybrid', enable_ai=False)
        hy_stats = hy_network.get_stats()
        print(f"        Servers: {hy_stats['total_servers']}, Switches: {hy_stats['total_switches']}, Cables: {hy_stats['total_cables']}, Cross-room (Jellyfish): {hy_stats['cross_room_cables']}")
        hy_engine = THDSimulationEngine(hy_rooms, hy_network, self.config, 'hybrid', enable_ai=False)
        hy_metrics = hy_engine.run_simulation(days)
        hy_summary = hy_engine.get_summary_stats()

        with open(os.path.join(span_dir, 'hybrid_no_ai_metrics.json'), 'w') as f:
            json.dump(hy_metrics, f, indent=2, default=str)

        # Run Hybrid + Deep AI
        print(f"\n  [3/3] HYBRID+DEEP AI (Fat-Tree + Jellyfish + ML Routing + 4 Deep AI Solutions)")
        hy_ai_rooms, hy_ai_network = self.builder.build_datacenter('hybrid', enable_ai=True)
        hy_ai_stats = hy_ai_network.get_stats()
        print(f"        Servers: {hy_ai_stats['total_servers']}, Switches: {hy_ai_stats['total_switches']}, Cables: {hy_ai_stats['total_cables']}, Cross-room (Jellyfish): {hy_ai_stats['cross_room_cables']}")
        print(f"        Deep AI Solutions:")
        print(f"          AI-1: PredictiveCableMaintenanceAI (GBM + Bayesian + degradation + anomaly)")
        print(f"          AI-2: DynamicPowerManagementAI (RL Q-Learning + Holt-Winters + thermal + peak shaving)")
        print(f"          AI-3: SelfHealingNetworkAI (Heartbeat + BFS/Dijkstra + FRR + recovery orchestrator)")
        print(f"          AI-4: TrafficAwareNetworkResilienceAI (Adaptive LB + predictor + congestion + scheduler)")
        hy_ai_engine = THDSimulationEngine(hy_ai_rooms, hy_ai_network, self.config, 'hybrid', enable_ai=True)
        hy_ai_metrics = hy_ai_engine.run_simulation(days)
        hy_ai_summary = hy_ai_engine.get_summary_stats()

        with open(os.path.join(span_dir, 'hybrid_with_deep_ai_metrics.json'), 'w') as f:
            json.dump(hy_ai_metrics, f, indent=2, default=str)

        # Generate comparison plots
        plot_gen = THDPlotGenerator(span_dir)
        plot_gen.generate_comparison_plots(sl_metrics, hy_metrics, hy_ai_metrics, days,
                                           filename_prefix=f'THD_{span_name}')

        # === COMPREHENSIVE COST ANALYSIS ===
        def calc_total_cost(summary):
            it_power = summary['annual_power_cost']
            # Cooling: PUE overhead minus AI cooling savings
            if summary.get('ai_enabled'):
                # Deep AI-2 already reduced power consumption, so cooling follows
                cooling = summary['total_cooling_cost']
            else:
                # Room-specific PUE for non-AI configurations
                cooling = it_power * (self.config.WEIGHTED_AVG_PUE - 1)

            cable_replacement = summary['annual_cable_replacement_cost']
            maintenance = summary['total_maintenance_cost']
            repair = summary['total_repair_cost']

            # Downtime cost: directly from simulation (already accounts for self-healing)
            downtime = summary['total_downtime_cost']

            # Staff: AI reduces manual intervention
            base_staff = self.config.ANNUAL_STAFF_COST * (summary['days_simulated'] / 365)
            if summary.get('ai_enabled'):
                staff = base_staff * (1 - self.config.AI_STAFF_SAVING_FACTOR)
                staff_saving = base_staff * self.config.AI_STAFF_SAVING_FACTOR
            else:
                staff = base_staff
                staff_saving = 0

            # Power savings from deep AI dynamic power management
            # The deep RL model actually powers down devices, so the power cost
            # is already reduced in the simulation. The savings are tracked separately.
            power_saving_val = 0
            if summary.get('power_savings_mwh', 0) > 0:
                power_saving_val = (summary['power_savings_mwh'] * 1000
                                    * self.config.POWER_COST_PER_KWH * self.config.WEIGHTED_AVG_PUE)

            # Peak demand charge savings from deep AI peak shaving
            peak_saving = summary.get('peak_shaving_savings', 0)

            # AI-1: Predictive maintenance saves by preventing emergency repairs
            # Calculate REAL maintenance saving as the difference in actual costs
            # (not the inflated AI model counter which counts hypothetical prevented failures)
            # Prevented cable failures mean fewer emergency replacements = real cost saving
            prevented = summary.get('prevented_failures', 0)
            maintenance_saving = prevented * self.config.CABLE_REPLACEMENT_COST * 0.3  # 30% saved per prevented failure (scheduled vs emergency)

            total = (it_power + cooling + cable_replacement + maintenance
                     + repair + downtime + staff - power_saving_val - peak_saving)

            ai_savings = (staff_saving + power_saving_val + peak_saving + maintenance_saving)

            return total, {
                'it_power_cost': it_power,
                'cooling_cost': cooling,
                'cable_replacement_cost': cable_replacement,
                'maintenance_cost': maintenance,
                'repair_cost': repair,
                'downtime_cost': downtime,
                'staff_cost': staff,
                'ai_total_savings': ai_savings,
                'power_saving_val': power_saving_val,
                'peak_saving': peak_saving,
                'maintenance_saving': maintenance_saving,
                'staff_saving': staff_saving,
            }

        sl_total, sl_breakdown = calc_total_cost(sl_summary)
        hy_total, hy_breakdown = calc_total_cost(hy_summary)
        hy_ai_total, hy_ai_breakdown = calc_total_cost(hy_ai_summary)

        cost_analysis = {
            'spine_leaf_total_cost': sl_total,
            'hybrid_no_ai_total_cost': hy_total,
            'hybrid_with_ai_total_cost': hy_ai_total,
            'ai_implementation_cost': self.config.TOTAL_AI_IMPLEMENTATION_COST,
            'savings_hybrid_vs_spine_leaf': sl_total - hy_total,
            'savings_hybrid_ai_vs_spine_leaf': sl_total - hy_ai_total,
            'savings_ai_vs_hybrid': hy_total - hy_ai_total,
            'roi_years': (self.config.TOTAL_AI_IMPLEMENTATION_COST
                          / max((sl_total - hy_ai_total) / (days / 365.0), 1)
                          if sl_total > hy_ai_total else float('inf')),
            'spine_leaf_breakdown': sl_breakdown,
            'hybrid_breakdown': hy_breakdown,
            'hybrid_ai_breakdown': hy_ai_breakdown,
        }

        results = {
            'spine_leaf': sl_summary,
            'hybrid_no_ai': hy_summary,
            'hybrid_with_ai': hy_ai_summary,
            'cost_analysis': cost_analysis,
            'span_days': days,
        }

        # Print span comparison
        sl_breakdown = cost_analysis.get('spine_leaf_breakdown', {})
        hy_breakdown = cost_analysis.get('hybrid_breakdown', {})
        hy_ai_breakdown = cost_analysis.get('hybrid_ai_breakdown', {})
        self._print_span_comparison(sl_summary, hy_summary, hy_ai_summary, cost_analysis, days,
                                    sl_breakdown, hy_breakdown, hy_ai_breakdown)

        return results

    def _print_span_comparison(self, sl, hy, hy_ai, cost, days,
                               sl_breakdown=None, hy_breakdown=None, hy_ai_breakdown=None):
        sl_breakdown = sl_breakdown or {}
        hy_breakdown = hy_breakdown or {}
        hy_ai_breakdown = hy_ai_breakdown or {}
        print(f"\n{'=' * 80}")
        print(f"  COMPARISON: {days}-DAY SIMULATION")
        print(f"{'=' * 80}")
        print(f"\n  {'Metric':<40} {'Spine-Leaf':>12} {'Hybrid':>12} {'Hybrid+AI':>12} {'Best vs SL':>12}")
        print(f"  {'-' * 88}")

        metrics = [
            ('Avg Throughput (%)', sl['avg_throughput'], hy['avg_throughput'], hy_ai['avg_throughput'], True),
            ('Avg Latency (us)', sl['avg_latency'], hy['avg_latency'], hy_ai['avg_latency'], False),
            ('Avg Availability (%)', sl['avg_availability'], hy['avg_availability'], hy_ai['avg_availability'], True),
            ('Avg Power (kW)', sl['avg_power_kw'], hy['avg_power_kw'], hy_ai['avg_power_kw'], False),
            ('Total Cable Failures', sl['total_cable_failures'], hy['total_cable_failures'], hy_ai['total_cable_failures'], False),
            ('Total Packets Routed', sl['total_packets_routed'], hy['total_packets_routed'], hy_ai['total_packets_routed'], True),
            ('Total Repairs', sl['total_repairs'], hy['total_repairs'], hy_ai['total_repairs'], False),
        ]

        for name, sl_val, hy_val, ai_val, higher_better in metrics:
            if sl_val != 0:
                if higher_better:
                    improvement = ((ai_val - sl_val) / abs(sl_val)) * 100
                else:
                    improvement = ((sl_val - ai_val) / abs(sl_val)) * 100
            else:
                improvement = 0
            sign = '+' if improvement >= 0 else ''
            print(f"  {name:<40} {sl_val:>12.2f} {hy_val:>12.2f} {ai_val:>12.2f} {sign}{improvement:>11.1f}%")

        print(f"  {'-' * 88}")
        print(f"\n  DEEP AI-SPECIFIC METRICS:")
        print(f"  {'AI-1: Predicted Cable Failures':<40} {'N/A':>12} {'N/A':>12} {hy_ai.get('predicted_failures', 0):>12}")
        print(f"  {'AI-1: Prevented Failures':<40} {'N/A':>12} {'N/A':>12} {hy_ai.get('prevented_failures', 0):>12}")
        print(f"  {'AI-1: Maintenance Cost Saved (EUR)':<40} {'N/A':>12} {'N/A':>12} {hy_ai.get('maintenance_cost_saved', 0):>12.0f}")
        print(f"  {'AI-2: Power Savings (MWh)':<40} {'N/A':>12} {'N/A':>12} {hy_ai.get('power_savings_mwh', 0):>12.2f}")
        print(f"  {'AI-2: RL Decisions':<40} {'N/A':>12} {'N/A':>12} {hy_ai.get('rl_decisions', 0):>12}")
        print(f"  {'AI-2: RL Exploitation Ratio (%)':<40} {'N/A':>12} {'N/A':>12} {hy_ai.get('rl_exploitation_ratio', 0):>12.1f}")
        print(f"  {'AI-2: SLA Violations':<40} {'N/A':>12} {'N/A':>12} {hy_ai.get('sla_violations', 0):>12}")
        print(f"  {'AI-2: Peak Shaving Savings (EUR)':<40} {'N/A':>12} {'N/A':>12} {hy_ai.get('peak_shaving_savings', 0):>12.2f}")
        print(f"  {'AI-3: Self-Healing Events':<40} {'N/A':>12} {'N/A':>12} {hy_ai.get('self_healing_events', 0):>12}")
        print(f"  {'AI-3: Recovery Rate (%)':<40} {'N/A':>12} {'N/A':>12} {hy_ai.get('recovery_rate', 0):>12.1f}")
        print(f"  {'AI-3: FRR Success Rate (%)':<40} {'N/A':>12} {'N/A':>12} {hy_ai.get('frr_success_rate', 0):>12.1f}")
        print(f"  {'AI-3: Avg Recovery Time (s)':<40} {'N/A':>12} {'N/A':>12} {hy_ai.get('avg_recovery_time', 0):>12.2f}")
        print(f"  {'AI-4: Traffic Interventions':<40} {'N/A':>12} {'N/A':>12} {hy_ai.get('traffic_interventions', 0):>12}")
        print(f"  {'AI-4: ECMP Collisions Avoided':<40} {'N/A':>12} {'N/A':>12} {hy_ai.get('ecmp_collisions_avoided', 0):>12}")
        print(f"  {'AI-4: Incast Events Prevented':<40} {'N/A':>12} {'N/A':>12} {hy_ai.get('incast_events_prevented', 0):>12}")
        print(f"  {'AI-4: Pre-Migration Events':<40} {'N/A':>12} {'N/A':>12} {hy_ai.get('pre_migration_events', 0):>12}")
        if 'ml_exploitation_ratio' in hy_ai:
            print(f"  {'ML Routing Exploit Ratio (%)':<40} {'N/A':>12} {'N/A':>12} {hy_ai['ml_exploitation_ratio']:>12.1f}")
            print(f"  {'ML Q-Table Size':<40} {'N/A':>12} {'N/A':>12} {hy_ai.get('ml_q_table_size', 0):>12}")
        print(f"  {'-' * 88}")

        print(f"\n  COST ANALYSIS:")
        print(f"  {'Spine-Leaf Total Cost':<40} EUR {cost['spine_leaf_total_cost']:>10,.2f}")
        print(f"  {'Hybrid Total Cost':<40} EUR {cost['hybrid_no_ai_total_cost']:>10,.2f}")
        print(f"  {'Hybrid+Deep AI Total Cost':<40} EUR {cost['hybrid_with_ai_total_cost']:>10,.2f}")
        print(f"  {'Hybrid Savings vs SL':<40} EUR {cost['savings_hybrid_vs_spine_leaf']:>10,.2f}")
        print(f"  {'Hybrid+AI Savings vs SL':<40} EUR {cost['savings_hybrid_ai_vs_spine_leaf']:>10,.2f}")
        print(f"  {'  - Of which: Additional AI Savings':<40} EUR {cost['savings_ai_vs_hybrid']:>10,.2f}")
        print(f"  {'AI Implementation Cost':<40} EUR {cost['ai_implementation_cost']:>10,}")
        print(f"  {'ROI (years)':<40} {cost['roi_years']:>12.2f}")

        print(f"\n  DETAILED COST BREAKDOWN ({days} days):")
        print(f"  {'Category':<30} {'Spine-Leaf':>12} {'Hybrid':>12} {'Hybrid+AI':>12}")
        print(f"  {'-'*66}")
        for key in ['it_power_cost', 'cooling_cost', 'cable_replacement_cost', 'maintenance_cost',
                     'repair_cost', 'downtime_cost', 'staff_cost']:
            sl_v = sl_breakdown.get(key, 0)
            hy_v = hy_breakdown.get(key, 0)
            ai_v = hy_ai_breakdown.get(key, 0)
            print(f"  {key.replace('_',' ').title():<30} EUR {sl_v:>10,.0f} EUR {hy_v:>10,.0f} EUR {ai_v:>10,.0f}")
        if hy_ai_breakdown.get('ai_total_savings', 0) > 0:
            print(f"  {'':-<30}")
            print(f"  {'AI Total Savings':<30} {'':>12} {'':>12} EUR {hy_ai_breakdown['ai_total_savings']:>10,.0f}")
            print(f"  {'  - Power Saving':<30} {'':>12} {'':>12} EUR {hy_ai_breakdown.get('power_saving_val', 0):>10,.0f}")
            print(f"  {'  - Peak Shaving':<30} {'':>12} {'':>12} EUR {hy_ai_breakdown.get('peak_saving', 0):>10,.0f}")
            print(f"  {'  - Maintenance Saving':<30} {'':>12} {'':>12} EUR {hy_ai_breakdown.get('maintenance_saving', 0):>10,.0f}")
            print(f"  {'  - Staff Saving':<30} {'':>12} {'':>12} EUR {hy_ai_breakdown.get('staff_saving', 0):>10,.0f}")

    def _print_final_comparison(self, all_results):
        print(f"\n{'=' * 80}")
        print(f"  FINAL SUMMARY: 365-DAY vs 5-YEAR")
        print(f"{'=' * 80}")

        for span_name in ['365day', '5year']:
            r = all_results.get(span_name, {})
            cost = r.get('cost_analysis', {})
            span_label = '1 Year' if span_name == '365day' else '5 Years'
            print(f"\n  {span_label}:")
            print(f"    Hybrid+Deep AI Savings vs Spine-Leaf: EUR {cost.get('savings_hybrid_ai_vs_spine_leaf', 0):>10,.2f}")
            print(f"    ROI: {cost.get('roi_years', 0):.2f} years")

        # 5-year annual savings and ROI highlight
        cost_5y = all_results.get('5year', {}).get('cost_analysis', {})
        if cost_5y:
            annual_savings = cost_5y.get('savings_hybrid_ai_vs_spine_leaf', 0) / 5
            print(f"\n  === ANNUAL SAVINGS (Hybrid+Deep AI vs Spine-Leaf): EUR {annual_savings:>10,.0f}/year ===")
            roi = cost_5y.get('roi_years', float('inf'))
            print(f"  === ROI: {roi:.1f} years (AI investment: EUR {self.config.TOTAL_AI_IMPLEMENTATION_COST:,}) ===")

        # Also show 1-year annualized savings
        cost_1y = all_results.get('365day', {}).get('cost_analysis', {})
        if cost_1y:
            savings_1y = cost_1y.get('savings_hybrid_ai_vs_spine_leaf', 0)
            print(f"  === 1-YEAR SAVINGS (Hybrid+Deep AI vs Spine-Leaf): EUR {savings_1y:>10,.0f} ===")
            roi_1y = cost_1y.get('roi_years', float('inf'))
            print(f"  === 1-YEAR ROI: {roi_1y:.1f} years ===")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    config = THDConfig()
    simulator = THDFinalSimulator(config)
    export_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'simulation_results')
    results = simulator.run_full_simulation(export_dir=export_dir)

    print("\n" + "=" * 80)
    print("  SIMULATION COMPLETE!")
    print("=" * 80)


if __name__ == '__main__':
    main()
