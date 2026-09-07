import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from ..core.fault_frequencies import FaultFrequencies
from ..analysis.filter_search_results import FilterSearchResults


_SHORT_FAULT_NAMES = {
    "f_cage": "c",
    "f_rolling_element": "re",
    "f_inner_ring": "ir",
    "f_outer_ring": "or",
    "f_rolling_element_about_axis": "reax",
}

_SIDEBAND_COLORS = (
    "#d55e00",
    "#0072b2",
    "#009e73",
    "#cc79a7",
    "#e69f00",
)

_HARMONIC_COLOR = "#4d4d4d"
_FAULT_FREQUENCY_COLOR = "#e41a1c"

def _short_fault_name(name: str) -> str:
    """
    Return the abbreviated label for a bearing fault frequency.

    parameters
    ----------
    name : str
        Full fault-frequency name.

    returns
    -------
    str
        Abbreviated name when one is defined, otherwise ``name`` unchanged.
    """
    return _SHORT_FAULT_NAMES.get(name, name)

def _scaled_fault_frequency_items(
        fault_frequencies: FaultFrequencies,
        f_of_rotation: float,
    ) -> list[tuple[str, float]]:
    """
    Return fault-frequency names and values scaled to a rotation frequency.

    parameters
    ----------
    fault_frequencies : FaultFrequencies
        Fault frequencies and their reference rotation frequency.
    f_of_rotation : float
        Rotation frequency to which the fault frequencies are scaled.

    returns
    -------
    list[tuple[str, float]]
        Fault-frequency names paired with values at ``f_of_rotation``.
    """
    frequency_items = list(fault_frequencies.fault_frequencies().items())
    reference_rotation = fault_frequencies.f_of_rotation

    if f_of_rotation != reference_rotation:
        print("Warning: The provided rotation frequency does not match the one from the fault frequencies. Transforming the fault frequencies to match the provided rotation frequency.")
        scale = f_of_rotation / reference_rotation
        frequency_items = [
            (name, value * scale)
            for name, value in frequency_items
        ]

    return frequency_items

def _add_visibility_buttons(
        figure: go.Figure,
        group_name: str,
        shape_indices: list[int],
        annotation_indices: list[int],
    ) -> go.Figure:
    """
    Add hide/show buttons for selected Plotly shapes and annotations.

    parameters
    ----------
    figure : go.Figure
        Figure whose layout receives the buttons.
    group_name : str
        Label used in the button names.
    shape_indices : list[int]
        Indices of shapes controlled by the buttons.
    annotation_indices : list[int]
        Indices of annotations controlled by the buttons.

    returns
    -------
    go.Figure
        The input figure with updated visibility buttons and layout.
    """
    hide_args = {
        f"shapes[{index}].visible": False
        for index in shape_indices
    }
    hide_args.update({
        f"annotations[{index}].visible": False
        for index in annotation_indices
    })

    show_args = {
        f"shapes[{index}].visible": True
        for index in shape_indices
    }
    show_args.update({
        f"annotations[{index}].visible": True
        for index in annotation_indices
    })

    existing_menus = list(figure.layout.updatemenus or ())
    menu_count = len(existing_menus) + 2
    group_count = (menu_count + 1) // 2
    row_spacing = 0.105
    menu_top = 0.92
    groups_per_row = 1 if group_count <= 2 else 2
    row_count = (group_count + groups_per_row - 1) // groups_per_row
    column_spacing = 0.48 if groups_per_row == 1 else 0.24
    menus = []

    for index, menu in enumerate(existing_menus):
        group_index = index // 2
        pair_column = group_index % groups_per_row
        column = index % 2
        menu = menu.to_plotly_json() if hasattr(menu, "to_plotly_json") else dict(menu)
        menu.update(
            x=0.02 + pair_column * 0.48 + column * 0.24,
            xanchor="left",
            yanchor="top",
        )
        menus.append(menu)

    button_specs = [
        (f"Hide {group_name}", hide_args),
        (f"Show {group_name}", show_args),
    ]
    new_group_index = len(existing_menus) // 2
    pair_column = new_group_index % groups_per_row
    for column, (label, args) in enumerate(button_specs):
        menus.append(dict(
            type="buttons",
            direction="right",
            x=0.02 + pair_column * 0.48 + column * column_spacing,
            xanchor="left",
            yanchor="top",
            pad={"r": 10, "t": 2, "b": 2},
            showactive=False,
            buttons=[dict(
                label=label,
                method="relayout",
                args=[args],
            )],
        ))

    plot_top = max(0.48, menu_top - (row_count - 1) * row_spacing - 0.12)
    for index, menu in enumerate(menus):
        group_index = index // 2
        menu["x"] = 0.02 + (group_index % groups_per_row) * 0.48 + (index % 2) * column_spacing
        menu["y"] = menu_top - (group_index // groups_per_row) * row_spacing

    for annotation in figure.layout.annotations:
        if annotation.yref == "paper" and annotation.y != 0:
            annotation.y = plot_top + 0.015
            annotation.yshift = 0

    title = figure.layout.title
    title_kwargs = {"y": 0.99, "yanchor": "top"}
    if title is not None and title.text is not None:
        title_kwargs["text"] = title.text

    figure.update_layout(
        height=figure.layout.height or 620,
        margin={"t": 30, "b": 95, "l": 60, "r": 30},
        yaxis={"domain": [0, plot_top]},
        title=title_kwargs,
    )
    figure.layout.updatemenus = menus

    return figure

def add_vlines_with_visibility_buttons(
        figure: go.Figure,
        x_values: list[float] | np.ndarray,
        group_name: str,
        labels: list[str] | None = None,
        line_dash: str = "dash",
        line_color: str | None = None,
        annotation_y: float = 1,
        annotation_yshift: int = 8,
    ) -> go.Figure:
    """
    Add vertical lines and buttons to control their visibility.

    parameters
    ----------
    figure : go.Figure
        Figure to which the vertical lines and controls are added.
    x_values : list[float] | np.ndarray
        X-coordinates of the vertical lines.
    group_name : str
        Label used in the hide/show button names.
    labels : list[str] | None, optional
        Labels displayed next to the lines, by default ``None``.
    line_dash : str, optional
        Plotly dash style for the lines, by default ``"dash"``.
    line_color : str | None, optional
        Line color, by default ``None``.
    annotation_y : float, optional
        Annotation position in paper coordinates, by default ``1``.
    annotation_yshift : int, optional
        Annotation pixel offset, by default ``8``.

    returns
    -------
    go.Figure
        The input figure with vertical lines, annotations, and controls.
    """
    shape_indices = []
    annotation_indices = []

    for index, x_value in enumerate(x_values):
        shape_index = len(figure.layout.shapes)
        line_kwargs = {
            "x": x_value,
            "line_dash": line_dash,
        }
        if line_color is not None:
            line_kwargs["line_color"] = line_color
        figure.add_vline(**line_kwargs)
        shape_indices.append(shape_index)

        if labels is not None:
            annotation_index = len(figure.layout.annotations)
            figure.add_annotation(
                x=x_value,
                y=annotation_y,
                xref="x",
                yref="paper",
                text=labels[index],
                showarrow=False,
                yshift=annotation_yshift,
            )
            annotation_indices.append(annotation_index)

    return _add_visibility_buttons(
        figure,
        group_name,
        shape_indices,
        annotation_indices,
    )

def plot_envelope(x:np.ndarray, envelope:np.ndarray, fs:int) -> go.Figure:
    """
    Plot a measured signal together with its envelope.

    parameters
    ----------
    x : np.ndarray
        Measured signal.
    envelope : np.ndarray
        Extracted envelope with the same length as ``x``.
    fs : int
        Sampling frequency in Hz.

    returns
    -------
    go.Figure
        Interactive Plotly figure containing the signal and envelope.
    """
    time = np.arange(len(x)) / fs
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x = time,
            y = x,
            mode = 'lines',
            name = 'Measured signal'
        )
    )
    figure.add_trace(
        go.Scatter(
            x = time,
            y = envelope,
            mode = 'lines',
            name = 'Envelope'
        )
    )
    figure.update_layout(
        title = "Envelope of the signal",
        xaxis_title = "Time [s]",
        yaxis_title = "Amplitude",
    )

    return figure

def plot_envelope_spectrum(
        freq: np.ndarray, 
        ampl: np.ndarray, 
        f_of_rotation: float, 
        fault_frequencies: FaultFrequencies, 
        plot_fault_frequencies: bool = True,
        normalise: bool = False
    ) -> go.Figure:
    """
    Plot an envelope spectrum with bearing fault-frequency markers.

    parameters
    ----------
    freq : np.ndarray
        Frequency values of the spectrum.
    ampl : np.ndarray
        Spectrum amplitudes.
    f_of_rotation : float
        Rotation frequency used for scaling and display.
    fault_frequencies : FaultFrequencies
        Bearing fault frequencies and their reference rotation frequency.
    plot_fault_frequencies : bool, optional
        Whether to add fault-frequency lines and controls, by default ``True``.
    normalise : bool, optional
        Whether to express frequency relative to the rotation frequency, by default ``False``.

    returns
    -------
    go.Figure
        Interactive envelope-spectrum figure.
    """
    # Calculate upper_x_limit based on the maximum fault frequency and check if the provided rotation frequency
    # matches the one from the fault frequencies.
    fault_frequency_items = _scaled_fault_frequency_items(
        fault_frequencies,
        f_of_rotation,
    )
    vlines_x_value = np.array([value for _, value in fault_frequency_items])
    upper_x_limit = 1.1 * np.max(vlines_x_value)

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x = freq if not normalise else freq / f_of_rotation,
            y = np.abs(ampl),
            mode = 'lines',
            name = 'Envelope spectrum'
        )
    )
    figure.update_layout(title="Envelope spectrum")
    if plot_fault_frequencies:
        plot_values = vlines_x_value if not normalise else vlines_x_value / f_of_rotation
        add_vlines_with_visibility_buttons(
            figure,
            plot_values,
            "fault frequencies",
            labels=[_short_fault_name(name) for name, _ in fault_frequency_items],
            annotation_yshift=+30,
            line_color=_FAULT_FREQUENCY_COLOR,
        )

    figure.update_layout(
        xaxis_title = "Frequency [Hz]" if not normalise else "Frequency/frequency_of_rotation [/]",
        yaxis_title = "Amplitude",
    )
    figure.update_xaxes(range=[0, upper_x_limit if not normalise else upper_x_limit / f_of_rotation])
    figure.update_yaxes(range=[0, np.max(np.abs(ampl))])

    return figure

def plot_harmonics(
    figure: go.Figure,
    fundamental_freq: float | int,
    num_harmonics: int,
    normalise: bool = False,
) -> go.Figure:
    """
    Add rotational-frequency harmonics and visibility controls.

    parameters
    ----------
    figure : go.Figure
        Figure to which harmonic lines are added.
    fundamental_freq : float | int
        Fundamental rotation frequency.
    num_harmonics : int
        Number of harmonics to draw.
    normalise : bool, optional
        Whether to express harmonic positions relative to the rotation frequency,
        by default ``False``.

    returns
    -------
    go.Figure
        The input figure with harmonic lines and controls.
    """
    harmonic_values = [fundamental_freq * i for i in range(1, num_harmonics + 1)]
    if normalise:
        harmonic_values = [value / fundamental_freq for value in harmonic_values]
    harmonic_labels = [
        f"{i} · f<sub>rot</sub>"
        for i in range(1, num_harmonics + 1)
    ]
    add_vlines_with_visibility_buttons(
        figure,
        harmonic_values,
        "harmonics",
        labels=harmonic_labels,
        line_dash="dot",
        line_color=_HARMONIC_COLOR,
        annotation_y=0,
        annotation_yshift=-30,
    )

    return figure

def plot_bands(
        figure: go.Figure,
        fundamental_freq: float | int,
        fault_frequencies: FaultFrequencies,
        num_sidebands: int,
        normalise: bool = False,
    ) -> go.Figure:
    """
    Add colored sideband lines and visibility controls.

    parameters
    ----------
    figure : go.Figure
        Figure to which sideband lines and controls are added.
    fundamental_freq : float | int
        Fundamental rotation frequency.
    fault_frequencies : FaultFrequencies
        Bearing fault frequencies and their reference rotation frequency.
    num_sidebands : int
        Number of lines added on each side of every fault frequency.
    normalise : bool, optional
        Whether to express sideband positions relative to the rotation frequency, by default ``False``.

    returns
    -------
    go.Figure
        The input figure with colored sidebands, legend entries, and controls.
    """
    if num_sidebands < 0:
        raise ValueError("num_sidebands must be non-negative")

    frequency_items = _scaled_fault_frequency_items(
        fault_frequencies,
        fundamental_freq,
    )
    all_sideband_shape_indices = []

    for index, (name, fault_frequency) in enumerate(frequency_items):
        short_name = _short_fault_name(name)
        line_color = _SIDEBAND_COLORS[index % len(_SIDEBAND_COLORS)]
        sideband_values = []
        for sideband_number in range(1, num_sidebands + 1):
            offset = sideband_number * fundamental_freq
            lower_sideband = fault_frequency - offset
            upper_sideband = fault_frequency + offset

            if lower_sideband >= 0:
                sideband_values.append(lower_sideband)
            sideband_values.append(upper_sideband)

        if normalise:
            sideband_values = [value / fundamental_freq for value in sideband_values]

        first_shape_index = len(figure.layout.shapes)
        add_vlines_with_visibility_buttons(
            figure,
            sideband_values,
            f"{short_name} sidebands",
            line_dash="dashdot",
            line_color=line_color,
        )
        all_sideband_shape_indices.extend(
            range(first_shape_index, first_shape_index + len(sideband_values))
        )
        figure.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name=f"{short_name} sidebands",
                line={"color": line_color, "dash": "dashdot"},
                legendgroup=f"{name} sidebands",
                hoverinfo="skip",
            )
        )

    _add_visibility_buttons(
        figure,
        "all sidebands",
        all_sideband_shape_indices,
        [],
    )
    figure.update_layout(
        legend={
            "orientation": "h",
            "x": 0,
            "y": -0.18,
            "xanchor": "left",
            "yanchor": "top",
        },
    )

    return figure

def plot_filter_search_for_envelope_extraction(
        filter_search_results:FilterSearchResults,
        plot_harmonics_score:bool = True,
        plot_kurtosis_score:bool = False
    ):
    """
    Plot filter-search scores for envelope extraction.

    parameters
    ----------
    filter_search_results : FilterSearchResults
        Filter-search results containing filter bands and scores.
    plot_harmonics_score : bool, optional
        Whether to plot harmonic scores when available, by default ``True``.
    plot_kurtosis_score : bool, optional
        Whether to plot kurtosis scores when available, by default ``False``.

    returns
    -------
    None
        Displays one Matplotlib figure for each selected score.
    """
    names_for_plot = []
    scores_for_plot = []
    if plot_harmonics_score and filter_search_results.harmonics_score is not None:
        fault_types = list(filter_search_results.harmonics_score.__annotations__.keys())
        names_for_plot += fault_types
        scores_for_plot.extend(filter_search_results.harmonics_score.__getattribute__(key) for key in fault_types)
    elif plot_harmonics_score and filter_search_results.harmonics_score is None:
        print("Warning: plot_harmonics_score is True but harmonics_score is missing.")
    
    if plot_kurtosis_score and filter_search_results.kurtosis_score is not None:
        names_for_plot.append("kurtosis")
        scores_for_plot.append(filter_search_results.kurtosis_score)
    elif plot_kurtosis_score and filter_search_results.kurtosis_score is None:
        print("Warning: plot_kurtosis_score is True but kurtosis_score is missing.")

    for name, plot_score in zip(names_for_plot, scores_for_plot):
        levels = sorted(set(filter_search_results.level))

        # map actual level -> row number
        level_to_row = {lvl: i for i, lvl in enumerate(levels)}

        fig, ax = plt.subplots(figsize=(12, 6))

        norm = Normalize(vmin=min(plot_score), vmax=max(plot_score))
        cmap = plt.cm.viridis

        for f_low, f_high, level, score in zip(
            filter_search_results.f_low,
            filter_search_results.f_high,
            filter_search_results.level,
            plot_score
        ):

            row = level_to_row[level]

            rect = Rectangle(
                (f_low, row),
                f_high - f_low,
                1.0,
                facecolor=cmap(norm(score)),
                edgecolor="none"
            )

            ax.add_patch(rect)

        ax.relim()
        ax.autoscale_view()
        ax.set_ylim(len(levels), 0)
        ax.set_xlim(min(filter_search_results.f_low), max(filter_search_results.f_high))
        ax.set_yticks(np.arange(len(levels)) + 0.5)
        ax.set_yticklabels([f"{lvl:.1f}" for lvl in levels])

        ax.set_ylabel("Level")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_title(f"{str(name).replace('_', ' ').title()}")

        sm = ScalarMappable(norm=norm, cmap=cmap) 
        plt.colorbar(sm, ax=ax, label="Score")
        plt.show()
