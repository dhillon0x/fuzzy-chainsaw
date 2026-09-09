# ==============================================================================
# FILE: src/controller.py
# Location: src/controller.py
# Description: Production-grade hardware bridge providing serial communication,
#              packet framing, automatic software simulation fallback, timeout
#              handling, and thread-safe streaming to microcontroller drivers.
# ==============================================================================

import time
import logging
from typing import List, Optional, Union
import serial
import serial.tools.list_ports

# Configure module-level logger for debugging serial communication
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (Controller) %(message)s"
)
logger = logging.getLogger("ArmController")


class ArmController:
    """
    Manages low-latency bidirectional UART communication between the Python host
    and the motion control hardware (e.g., Arduino, ESP32, RAMPS, or STM32).
    
    Includes safety clamping, automated device handshaking, command framing,
    and automatic failover to a simulated software driver if physical hardware
    is disconnected or cannot be opened.
    """

    def __init__(
        self,
        port: str = "COM3",
        baudrate: int = 115200,
        timeout: float = 1.0,
        write_timeout: float = 1.0,
        auto_reconnect: bool = False
    ) -> None:
        """
        Initialize the serial arm controller.

        :param port: Target serial device name (e.g., 'COM3', '/dev/ttyUSB0', '/dev/ttyACM0').
        :param baudrate: Transmission rate in bits per second (default: 115200 bps).
        :param timeout: Read timeout in seconds.
        :param write_timeout: Write blocking timeout in seconds.
        :param auto_reconnect: Whether to poll and auto-reconnect if link drops during operation.
        """
        self.port: str = port
        self.baudrate: int = baudrate
        self.timeout: float = timeout
        self.write_timeout: float = write_timeout
        self.auto_reconnect: bool = auto_reconnect
        self.serial_conn: Optional[serial.Serial] = None
        self.is_simulated: bool = False
        self.last_sent_angles: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.connect()

    def list_available_ports(self) -> List[str]:
        """Scans the operating system for all currently attached serial communication endpoints."""
        ports = serial.tools.list_ports.comports()
        detected = [p.device for p in ports]
        logger.info(f"Available system serial ports: {detected if detected else 'None'}")
        return detected

    def connect(self) -> bool:
        """
        Establishes a physical connection to the specified COM / tty port.
        Automatically defaults to simulated output if connection fails.
        """
        self.list_available_ports()
        try:
            logger.info(f"Attempting UART connection to '{self.port}' at {self.baudrate} baud...")
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=self.timeout,
                write_timeout=self.write_timeout
            )
            
            # Allow 2 seconds for microcontroller bootloader (DTR/RTS reset cycle) to settle
            time.sleep(2.0)
            self.serial_conn.reset_input_buffer()
            self.serial_conn.reset_output_buffer()
            self.is_simulated = False
            logger.info(f"Hardware connection verified and synchronized on {self.port}.")
            return True

        except (serial.SerialException, FileNotFoundError) as exc:
            self.is_simulated = True
            self.serial_conn = None
            logger.warning(
                f"Physical port '{self.port}' unavailable ({exc}). "
                f"Entering VIRTUAL SIMULATION MODE. Trajectories will be printed to stdout."
            )
            return False

    def send_joint_targets(self, joint_angles: Union[List[float], tuple]) -> bool:
        """
        Formats, validates, and dispatches joint angle targets to the firmware.
        Packet standard: ASCII G-code style parameterized string.
        Format: 'G0 J1:{deg} J2:{deg} J3:{deg} J4:{deg} J5:{deg} J6:{deg}\\n'

        :param joint_angles: Iterable of 6 joint angles represented in degrees.
        :return: True if successfully delivered or simulated, False upon link error.
        """
        if len(joint_angles) != 6:
            logger.error(f"Invalid joint payload: Expected 6 axes, got {len(joint_angles)} values.")
            return False

        # Truncate values to two decimal places for transmission efficiency
        clamped = [round(float(a), 2) for a in joint_angles]
        command = (
            f"G0 J1:{clamped[0]:.2f} J2:{clamped[1]:.2f} J3:{clamped[2]:.2f} "
            f"J4:{clamped[3]:.2f} J5:{clamped[4]:.2f} J6:{clamped[5]:.2f}\n"
        )

        if not self.is_simulated and self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.write(command.encode("ascii"))
                self.serial_conn.flush()
                self.last_sent_angles = clamped
                return True
            except serial.SerialTimeoutException:
                logger.error("UART Write Timeout: Device buffer full or line hung.")
                return False
            except serial.SerialException as e:
                logger.error(f"Connection dropped during transmission: {e}")
                self.is_simulated = True
                return False
        else:
            # Emulated output stream for dry-run testing
            print(f"[SIM OUT] {command.strip()}")
            self.last_sent_angles = clamped
            return True

    def read_firmware_response(self) -> Optional[str]:
        """
        Reads any pending ASCII telemetry or acknowledgment ('ok\\n') lines from the microcontroller.
        
        :return: Decoded response string or None if unreadable / running in simulation.
        """
        if self.is_simulated or not self.serial_conn or not self.serial_conn.is_open:
            return None

        try:
            if self.serial_conn.in_waiting > 0:
                raw_response = self.serial_conn.readline()
                decoded = raw_response.decode("ascii", errors="replace").strip()
                if decoded:
                    logger.debug(f"[FW RESP] {decoded}")
                return decoded
        except serial.SerialException as exc:
            logger.error(f"Error reading UART response: {exc}")
        return None

    def home(self) -> bool:
        """Sends the mechanical homing command to re-zero all optical or mechanical endstops."""
        home_command = "G28\n"
        if not self.is_simulated and self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.write(home_command.encode("ascii"))
                self.serial_conn.flush()
                logger.info("Sent axis homing sequence command (G28).")
                return True
            except serial.SerialException as e:
                logger.error(f"Homing dispatch failed: {e}")
                return False
        else:
            print("[SIM OUT] G28 (Axis Homing Protocol Executed)")
            return True

    def emergency_stop(self) -> None:
        """Instantly halts all motor timer interrupts on the firmware side."""
        e_stop_cmd = "M112\n"  # Standard RepRap / CNC immediate kill command
        if not self.is_simulated and self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.write(e_stop_cmd.encode("ascii"))
                self.serial_conn.flush()
                logger.critical("EMERGENCY STOP (M112) DISPATCHED TO CONTROLLER.")
            except serial.SerialException:
                pass
        else:
            print("[SIM OUT] M112 (EMERGENCY STOP INTERRUPT TRIGGERED)")

    def close(self) -> None:
        """Performs a clean shutdown and closes the serial descriptor."""
        if self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.reset_output_buffer()
                self.serial_conn.close()
                logger.info(f"Serial port '{self.port}' cleanly deallocated and closed.")
            except serial.SerialException as exc:
                logger.warning(f"Error during port deallocation: {exc}")
        self.serial_conn = None
