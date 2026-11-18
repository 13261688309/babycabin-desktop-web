"""Identity-gated native H743 CDC observer, never a USB-UART DTR override."""

from .mcu_status import parse_status, parse_motion
from .telemetry import SerialTelemetry, TelemetryDecoder
from .voice_observation import parse_voice_rx
from .voice_control import parse_voice_control
from .vision_protocol import parse_vision_link, observation_fresh


def native_wiring_scope(*, native_board_confirmed=False, isolated_usb_confirmed=False,
                        voice_rx_harness_confirmed=False, actuators_disconnected_confirmed=False,
                        separate_usb_power_confirmed=False, dht11_3v3_harness_confirmed=False,
                        voice_and_dht11_harness_confirmed=False):
    flags = (native_board_confirmed, isolated_usb_confirmed, voice_rx_harness_confirmed,
             actuators_disconnected_confirmed, separate_usb_power_confirmed,
             dht11_3v3_harness_confirmed, voice_and_dht11_harness_confirmed)
    if any(type(value) is not bool for value in flags):
        raise ValueError('Confirmations must be explicit booleans')
    if native_board_confirmed:
        if (isolated_usb_confirmed and not any(flags[2:])):
            return 'isolated_usb'
        if (not isolated_usb_confirmed and voice_rx_harness_confirmed and
                actuators_disconnected_confirmed and separate_usb_power_confirmed and
                not dht11_3v3_harness_confirmed and not voice_and_dht11_harness_confirmed):
            return 'controlled_powered_voice_rx'
        if (not isolated_usb_confirmed and dht11_3v3_harness_confirmed and
                actuators_disconnected_confirmed and not voice_rx_harness_confirmed and
                not separate_usb_power_confirmed and not voice_and_dht11_harness_confirmed):
            return 'usb_powered_dht11_3v3'
        if (voice_and_dht11_harness_confirmed and actuators_disconnected_confirmed and
                separate_usb_power_confirmed and not isolated_usb_confirmed and
                not voice_rx_harness_confirmed and not dht11_3v3_harness_confirmed):
            return 'controlled_powered_voice_rx_and_dht11_3v3'
    raise ValueError('Fresh native-board and exactly one complete wiring scope required')


class NativeStatusDecoder(TelemetryDecoder):
    def _decode(self, line):
        if line.startswith(b"voice_cmd "):
            try:
                return {"type":"voiceControl","data":parse_voice_control(line)}
            except (ValueError,UnicodeError):
                self.rejected+=1
                return None
        if line.startswith(b"motion "):
            try:
                return {"type": "motion", "data": parse_motion(line + b"\n")}
            except (ValueError, UnicodeError):
                self.rejected += 1
                return None
        if line.startswith(b"vision_link "):
            try:
                return {"type": "visionLink", "data": parse_vision_link(line)}
            except (ValueError, UnicodeError):
                self.rejected += 1
                return None
        if line.startswith(b"voice_rx "):
            try:
                return {"type": "voiceRx", "data": parse_voice_rx(line)}
            except (ValueError, UnicodeError):
                self.rejected += 1
                return None
        if not line.startswith(b"status "):
            return super()._decode(line)
        try:
            return {"type": "status", "data": parse_status(line + b"\n")}
        except (ValueError, UnicodeError):
            self.rejected += 1
            return None


class NativeUsbTelemetry(SerialTelemetry):
    def __init__(self, port: str, baud: int = 115200, stale_ms: int = 6000, *,
                 expected_serial: str, expected_location: str,
                 native_board_confirmed: bool = False, isolated_usb_confirmed: bool = False,
                 voice_rx_harness_confirmed: bool = False,
                 actuators_disconnected_confirmed: bool = False,
                 separate_usb_power_confirmed: bool = False,
                 dht11_3v3_harness_confirmed: bool = False,
                 voice_and_dht11_harness_confirmed: bool = False,
                 reconnect_seconds: float = 0) -> None:
        self._wiring_scope = native_wiring_scope(
            native_board_confirmed=native_board_confirmed, isolated_usb_confirmed=isolated_usb_confirmed,
            voice_rx_harness_confirmed=voice_rx_harness_confirmed,
            actuators_disconnected_confirmed=actuators_disconnected_confirmed,
            separate_usb_power_confirmed=separate_usb_power_confirmed,
            dht11_3v3_harness_confirmed=dht11_3v3_harness_confirmed,
            voice_and_dht11_harness_confirmed=voice_and_dht11_harness_confirmed)
        if any(type(value) is not str or not value.strip()
               for value in (port, expected_serial, expected_location)):
            raise ValueError('Explicit port, USB serial number and USB location required')
        self._expected_serial = expected_serial
        self._expected_location = expected_location
        super().__init__(port, baud, stale_ms, reconnect_seconds)

    def _new_decoder(self):
        return NativeStatusDecoder()

    def _open_link(self):
        from tools.capture_mcu_usb import require_endpoint

        # Reuse the capture tool's VID/PID/serial/location rule for every reopen.
        require_endpoint(self._port, self._expected_serial, self._expected_location)
        link = self._serial_factory(port=None, baudrate=self._baud, timeout=.2, write_timeout=.2,
                                    xonxoff=False, rtscts=False, dsrdtr=False)
        try:
            link.dtr = True
            link.rts = False
            link.port = self._port
            require_endpoint(self._port, self._expected_serial, self._expected_location)
            link.open()
            return link
        except BaseException:
            try:
                link.close()
            except Exception:
                self._open_cleanup_failed = True
            raise

    def snapshot(self, now_ms: int) -> dict:
        result = super().snapshot(now_ms)
        result.update(transport='h743_native_usb_cdc_read_only', dtrRequested=True,
                      endpointMetadataIsIndependentPhysicalIdentityProof=False,
                      wiringScope=self._wiring_scope, physicalStopQualified=False,
                      externalUartElectricalQualified=False)
        if 'visionLink' in result:
            result['visionLink']['observationFresh'] = observation_fresh(result['visionLink'])
        return result
