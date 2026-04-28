import pandas as pd
import numpy as np
import time
import os
from datetime import datetime, timedelta


class ICUSensorSimulator:
    def __init__(self, patient_id="SIM-123", buffer_path=None):
        self.patient_id = patient_id
        # Starting normal values
        self.current_vitals = {
            "heart_rate": 80.0,
            "spo2": 98.0,
            "resp_rate": 16.0,
            "temp": 37.0,
            "mean_bp": 90.0,
        }
        self.buffer_path = buffer_path if buffer_path else "data/live_stream.csv"

        # Only create directories if we're not using /tmp
        if not self.buffer_path.startswith("/tmp"):
            os.makedirs(os.path.dirname(self.buffer_path), exist_ok=True)

        # Initialize buffer

    def prefill_history(self, hours=24):
        """Generates back-dated history to ensure the pipeline has enough data for windows."""
        print(f"Prefilling {hours} hours of history...")
        now = datetime.now()
        data = []
        for h in range(hours * 12):  # 5-min intervals
            timestamp = (now - timedelta(minutes=h * 5)).strftime("%Y-%m-%d %H:%M:%S")
            self.simulate_drift()
            for vital, val in self.current_vitals.items():
                data.append(
                    {
                        "timestamp": timestamp,
                        "vital_name": vital,
                        "valuenum": round(float(val), 2),
                        "subject_id": self.patient_id,
                    }
                )

        df_hist = pd.DataFrame(data).sort_values("timestamp")
        df_hist.to_csv(self.buffer_path, index=False)
        print("Prefill complete.")

    def simulate_drift(self):
        """Simulates physiological drift with random walk."""
        drifts = {
            "heart_rate": np.random.normal(0, 2),
            "spo2": np.random.normal(-0.1, 0.2),  # Slight downward bias
            "resp_rate": np.random.normal(0, 0.5),
            "temp": np.random.normal(0, 0.05),
            "mean_bp": np.random.normal(0, 3),
        }

        for vital, drift in drifts.items():
            self.current_vitals[vital] += drift

            # Keep within semi-realistic clinical bounds
            if vital == "spo2":
                self.current_vitals[vital] = np.clip(
                    self.current_vitals[vital], 80, 100
                )
            if vital == "heart_rate":
                self.current_vitals[vital] = np.clip(
                    self.current_vitals[vital], 40, 180
                )
            if vital == "temp":
                self.current_vitals[vital] = np.clip(self.current_vitals[vital], 35, 42)

    def log_to_stream(self):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_data = []
        for vital, val in self.current_vitals.items():
            new_data.append(
                {
                    "timestamp": timestamp,
                    "vital_name": vital,
                    "valuenum": round(float(val), 2),
                    "subject_id": self.patient_id,
                }
            )

        df_new = pd.DataFrame(new_data)
        df_new.to_csv(self.buffer_path, mode="a", header=False, index=False)
        print(
            f"[{timestamp}] Logged vitals for {self.patient_id}: HR={round(self.current_vitals['heart_rate'],1)} SpO2={round(self.current_vitals['spo2'],1)}"
        )

    def run(self, interval_sec=5):
        print(f"--- Starting Real-Time Sensor Simulation for {self.patient_id} ---")
        print(f"Streaming data to: {self.buffer_path}")
        try:
            while True:
                self.simulate_drift()
                self.log_to_stream()

                # Maintain buffer size (keep last 1000 lines approx)
                # To be efficient, we'd use a different storage, but for simulation CSV is fine
                time.sleep(interval_sec)
        except KeyboardInterrupt:
            print("\nSimulation stopped.")


if __name__ == "__main__":
    simulator = ICUSensorSimulator()
    simulator.run()
