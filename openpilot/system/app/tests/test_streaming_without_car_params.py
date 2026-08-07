from openpilot.common.params import Params
from openpilot.common.test import OpenpilotTestCase
from openpilot.system.app.methods import dispatcher


class TestStreamingWithoutCarParams(OpenpilotTestCase):
  def test_start_stream_without_car_params(self, mocker):
    params = Params()
    params.remove("CarParamsPersistent")
    params.put_bool("IsOffroad", False)
    post_stream_request = mocker.patch(
      "openpilot.system.webrtc.helpers.post_stream_request",
      return_value={"sdp": "answer", "type": "answer"},
    )

    assert dispatcher["startStream"]("offer", True) == {"sdp": "answer", "type": "answer"}
    assert post_stream_request.call_args.args[0].bridge_services_in == []
