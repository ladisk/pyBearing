import sys
sys.path.insert(0, ".")
sys.path.insert(0, "../")
sys.path.insert(0, "../..")

import numpy as np
import plotly.graph_objects as go
import pytest

from pybearing.core.fault_frequencies import FaultFrequencies
from pybearing.visualization.envelope_analysis import (
    add_envelope_spectrum_trace,
    plot_bands,
    plot_envelope,
    plot_envelope_spectrum,
    plot_harmonics,
)


FAULT_FREQUENCIES = FaultFrequencies(
    f_cage=2.0,
    f_rolling_element_about_axis=3.0,
    f_outer_ring=4.0,
    f_rolling_element=5.0,
    f_inner_ring=6.0,
    f_of_rotation=1.0,
)


def button_labels(figure):
    return [
        button["label"]
        for menu in figure.layout.updatemenus
        for button in menu["buttons"]
    ]


class TestEnvelopeVisualization:
    def test_add_envelope_spectrum_trace_normalises_frequency_and_uses_name(self):
        figure = go.Figure()
        figure.update_layout(meta={"normalise": True})
        frequency = np.array([2.0, 4.0])
        amplitude = np.array([-1.0, 3.0])

        result = add_envelope_spectrum_trace(
            figure,
            frequency,
            amplitude,
            f_of_rotation=2.0,
            name="Second spectrum",
        )

        assert result is figure
        assert result.data[0].name == "Second spectrum"
        assert np.allclose(result.data[0].x, [1.0, 2.0])
        assert np.allclose(result.data[0].y, [1.0, 3.0])

    def test_add_envelope_spectrum_trace_rejects_non_normalised_figure(self):
        figure = go.Figure()
        figure.update_layout(meta={"normalise": False})

        with pytest.raises(
            ValueError,
            match="add_envelope_spectrum_trace requires a normalised figure.",
        ):
            add_envelope_spectrum_trace(
                figure,
                np.array([1.0]),
                np.array([1.0]),
                f_of_rotation=2.0,
                name="Spectrum",
            )

    def test_add_envelope_spectrum_trace_can_use_secondary_yaxis(self):
        figure = go.Figure()
        figure.update_layout(meta={"normalise": True})

        add_envelope_spectrum_trace(
            figure,
            np.array([1.0, 2.0]),
            np.array([10.0, 20.0]),
            f_of_rotation=1.0,
            name="Large spectrum",
            plot_on_secondary_yaxis=True,
        )

        assert figure.data[0].name == "Large spectrum (right y-axis)"
        assert figure.data[0].yaxis == "y2"
        assert figure.layout.yaxis2.overlaying == "y"
        assert figure.layout.yaxis2.side == "right"

    def test_plot_envelope_creates_signal_and_envelope_traces(self):
        signal = np.array([1.0, 2.0, 3.0])
        envelope = np.array([1.5, 2.5, 3.5])

        figure = plot_envelope(signal, envelope, fs=2)

        assert isinstance(figure, go.Figure)
        assert [trace.name for trace in figure.data] == ["Measured signal", "Envelope"]
        assert np.allclose(figure.data[0].x, [0.0, 0.5, 1.0])
        assert np.array_equal(figure.data[1].y, envelope)

    def test_plot_envelope_spectrum_normalises_frequency_axis_and_fault_lines(self):
        frequency = np.array([0.0, 1.0, 2.0, 3.0])
        amplitude = np.ones(4)

        figure = plot_envelope_spectrum(
            frequency,
            amplitude,
            f_of_rotation=2.0,
            fault_frequencies=FAULT_FREQUENCIES,
            normalise=True,
        )

        assert figure.layout.meta == {"normalise": True}
        assert np.allclose(figure.data[0].x, frequency / 2.0)
        assert np.allclose(
            [shape.x0 for shape in figure.layout.shapes],
            list(FAULT_FREQUENCIES.fault_frequencies().values()),
        )
        assert all(shape.line.color == "#e41a1c" for shape in figure.layout.shapes)
        assert [annotation.text for annotation in figure.layout.annotations] == [
            "c", "reax", "or", "re", "ir"
        ]

    def test_plot_harmonics_normalises_positions(self):
        figure = plot_harmonics(
            go.Figure(),
            fundamental_freq=2.0,
            num_harmonics=3,
            normalise=True,
        )

        assert np.allclose(
            [shape.x0 for shape in figure.layout.shapes],
            [1.0, 2.0, 3.0],
        )
        assert all(shape.line.color == "#4d4d4d" for shape in figure.layout.shapes)
        assert [annotation.text for annotation in figure.layout.annotations] == [
            "1 · f<sub>rot</sub>",
            "2 · f<sub>rot</sub>",
            "3 · f<sub>rot</sub>",
        ]

    def test_plot_bands_adds_colored_legend_and_all_sidebands_controls(self):
        figure = plot_bands(
            go.Figure(),
            fundamental_freq=1.0,
            fault_frequencies=FAULT_FREQUENCIES,
            num_sidebands=1,
        )

        sideband_traces = figure.data
        assert [trace.name for trace in sideband_traces] == [
            "c sidebands",
            "reax sidebands",
            "or sidebands",
            "re sidebands",
            "ir sidebands",
        ]
        assert len({trace.line.color for trace in sideband_traces}) == 5
        assert all(trace.showlegend is None or trace.showlegend for trace in sideband_traces)

        labels = button_labels(figure)
        assert "Hide all sidebands" in labels
        assert "Show all sidebands" in labels
        assert "Hide c sidebands" in labels
        assert "Show ir sidebands" in labels

        all_sidebands_menu = next(
            menu
            for menu in figure.layout.updatemenus
            if menu.buttons[0]["label"] == "Hide all sidebands"
        )
        hide_args = all_sidebands_menu.buttons[0]["args"][0]
        assert len(hide_args) == 10
        assert set(hide_args) == {f"shapes[{index}].visible" for index in range(10)}

    def test_plot_bands_normalises_sideband_positions(self):
        figure = plot_bands(
            go.Figure(),
            fundamental_freq=2.0,
            fault_frequencies=FAULT_FREQUENCIES,
            num_sidebands=1,
            normalise=True,
        )

        assert np.allclose(
            [shape.x0 for shape in figure.layout.shapes],
            [1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0, 6.0, 5.0, 7.0],
        )

    def test_plot_bands_rejects_negative_sideband_count(self):
        with pytest.raises(ValueError, match="num_sidebands must be non-negative"):
            plot_bands(
                go.Figure(),
                fundamental_freq=1.0,
                fault_frequencies=FAULT_FREQUENCIES,
                num_sidebands=-1,
            )
