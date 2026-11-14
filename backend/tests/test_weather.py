"""任意地点天气实况、相对日期、指定日期和多日预报。"""

from unittest.mock import patch
import unittest

from langchain_core.tools import ToolException

from backend.app.agent.tools import get_weather


def forecast_payload():
    dates = [f"2026-11-{day:02d}" for day in range(13, 29)]
    count = len(dates)
    return {
        "current": {
            "weather_code": 0, "temperature_2m": 20, "apparent_temperature": 19,
            "relative_humidity_2m": 55, "wind_speed_10m": 8, "precipitation": 0,
        },
        "daily": {
            "time": dates, "weather_code": [61] * count,
            "temperature_2m_max": [22] * count, "temperature_2m_min": [12] * count,
            "apparent_temperature_max": [21] * count, "apparent_temperature_min": [11] * count,
            "precipitation_probability_max": [60] * count, "precipitation_sum": [3.5] * count,
            "wind_speed_10m_max": [18] * count,
        },
    }


class WeatherTests(unittest.TestCase):
    def responses(self, place="杭州"):
        return [{"results": [{"name": place, "latitude": 30.3, "longitude": 120.2}]}, forecast_payload()]

    def test_current_weather_remains_backward_compatible(self):
        with patch("backend.app.agent.tools._get_json", side_effect=self.responses()):
            report = get_weather.invoke({"location": "杭州"})
        self.assertIn("杭州当前天气：晴朗", report)
        self.assertIn("气温 20°C", report)

    def test_relative_and_multiday_forecast(self):
        with patch("backend.app.agent.tools._get_json", side_effect=self.responses()):
            report = get_weather.invoke({"location": "杭州", "date": "明天", "days": 3})
        self.assertEqual(report.count("杭州天气预报"), 3)
        self.assertIn("2026-11-14", report)
        self.assertIn("2026-11-16", report)
        self.assertIn("最大降水概率 60%", report)

    def test_explicit_date_and_arbitrary_place(self):
        with patch("backend.app.agent.tools._get_json", side_effect=self.responses("东京")):
            report = get_weather.invoke({"location": "东京", "date": "2026-11-20"})
        self.assertIn("东京天气预报（2026-11-20）", report)

    def test_rejects_unsupported_or_out_of_range_date(self):
        for requested in ("很久以后", "2027-01-01"):
            with self.subTest(requested=requested), patch(
                "backend.app.agent.tools._get_json", side_effect=self.responses()
            ), self.assertRaises(ToolException):
                get_weather.invoke({"location": "杭州", "date": requested})


if __name__ == "__main__":
    unittest.main()
