import json
from typing import Any, Dict, Optional

from dds.dds_base import DDSObject
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_


class DemoPoseDDS(DDSObject):
    """DDS node for demo root-pose command."""

    def __init__(self, node_name: str = "demo_pose_dds"):
        if hasattr(self, "_initialized"):
            return
        super().__init__()
        self._initialized = True
        self.node_name = node_name
        self.setup_shared_memory(
            output_shm_name="isaac_demo_pose_cmd",
            output_size=2048,
            outputshm_flag=True,
            inputshm_flag=False,
        )
        print(f"[{self.node_name}] Demo pose DDS node initialized")

    def setup_publisher(self) -> bool:
        return True

    def setup_subscriber(self) -> bool:
        try:
            self.subscriber = ChannelSubscriber("rt/demo_pose/cmd", String_)
            self.subscriber.Init(lambda msg: self.dds_subscriber(msg, ""), 1)
            print(f"[{self.node_name}] Demo pose command subscriber initialized")
            return True
        except Exception as e:
            print(f"[{self.node_name}] Failed to initialize demo pose subscriber: {e}")
            return False

    def dds_publisher(self) -> Any:
        return None

    def dds_subscriber(self, msg: String_, datatype: str = None) -> Dict[str, Any]:
        try:
            data = json.loads(msg.data)
            if isinstance(data, dict):
                self.output_shm.write_data(data)
                return data
        except Exception as e:
            print(f"[{self.node_name}] Failed to parse demo pose command: {e}")
        return {}

    def get_pose_command(self) -> Optional[Dict[str, Any]]:
        if self.output_shm:
            return self.output_shm.read_data()
        return None

    def clear(self):
        if self.output_shm:
            self.output_shm.write_data({"apply": 0})
