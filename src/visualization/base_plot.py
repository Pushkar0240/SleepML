"""
=========================================================
SleepTransitionFramework

Base Plot

Parent class for all visualization modules.

Responsibilities
----------------
* Figure creation
* Figure management
* Styling
* Axis utilities
* Labels
* Titles
* Grid
* Legends
* Tick formatting
=========================================================
"""

from __future__ import annotations

from abc import ABC
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


class BasePlot(ABC):

    """
    Parent class for all visualization classes.
    """

    # ---------------------------------------------------------

    def __init__(

        self,

        output_dir="outputs/plots",

        figsize=(10, 6),

        dpi=300,

        style="default",

        transparent=False

    ):

        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(

            parents=True,

            exist_ok=True

        )

        self.figsize = figsize

        self.dpi = dpi

        self.style = style

        self.transparent = transparent

        self.figure = None

        self.axis = None

    # ---------------------------------------------------------

    def create(

        self

    ):

        """
        Create a new figure.
        """

        plt.style.use(

            self.style

        )

        self.figure, self.axis = plt.subplots(

            figsize=self.figsize,

            dpi=self.dpi

        )

        return self.figure, self.axis

    # ---------------------------------------------------------

    def clear(self):

        """
        Clear current figure.
        """

        if self.figure is not None:

            self.figure.clf()

    # ---------------------------------------------------------

    def close(self):

        """
        Close figure.
        """

        if self.figure is not None:

            plt.close(

                self.figure

            )

    # ---------------------------------------------------------

    def title(

        self,

        text,

        fontsize=16,

        weight="bold"

    ):

        self.axis.set_title(

            text,

            fontsize=fontsize,

            fontweight=weight

        )

    # ---------------------------------------------------------

    def xlabel(

        self,

        label,

        fontsize=12

    ):

        self.axis.set_xlabel(

            label,

            fontsize=fontsize

        )

    # ---------------------------------------------------------

    def ylabel(

        self,

        label,

        fontsize=12

    ):

        self.axis.set_ylabel(

            label,

            fontsize=fontsize

        )

    # ---------------------------------------------------------

    def xlim(

        self,

        left,

        right

    ):

        self.axis.set_xlim(

            left,

            right

        )

    # ---------------------------------------------------------

    def ylim(

        self,

        bottom,

        top

    ):

        self.axis.set_ylim(

            bottom,

            top

        )

    # ---------------------------------------------------------

    def grid(

        self,

        enabled=True,

        alpha=0.3,

        linestyle="--"

    ):

        self.axis.grid(

            enabled,

            alpha=alpha,

            linestyle=linestyle

        )

    # ---------------------------------------------------------

    def legend(

        self,

        location="best",

        fontsize=10

    ):

        self.axis.legend(

            loc=location,

            fontsize=fontsize

        )

    # ---------------------------------------------------------

    def x_ticks(

        self,

        rotation=0,

        fontsize=10

    ):

        self.axis.tick_params(

            axis="x",

            labelrotation=rotation,

            labelsize=fontsize

        )

    # ---------------------------------------------------------

    def y_ticks(

        self,

        fontsize=10

    ):

        self.axis.tick_params(

            axis="y",

            labelsize=fontsize

        )

    # ---------------------------------------------------------

    def integer_x_axis(self):

        """
        Force integer values on x-axis.
        """

        self.axis.xaxis.set_major_locator(

            MaxNLocator(

                integer=True

            )

        )

    # ---------------------------------------------------------

    def integer_y_axis(self):

        self.axis.yaxis.set_major_locator(

            MaxNLocator(

                integer=True

            )

        )

    # ---------------------------------------------------------

    def horizontal_line(

        self,

        y,

        label=None,

        linestyle="--"

    ):

        self.axis.axhline(

            y,

            linestyle=linestyle,

            label=label

        )

    # ---------------------------------------------------------

    def vertical_line(

        self,

        x,

        label=None,

        linestyle="--"

    ):

        self.axis.axvline(

            x,

            linestyle=linestyle,

            label=label

        )

    # ---------------------------------------------------------

    def annotate(

        self,

        x,

        y,

        text

    ):

        self.axis.annotate(

            text,

            xy=(x, y)

        )

    # ---------------------------------------------------------

    def set_facecolor(

        self,

        color="white"

    ):

        self.axis.set_facecolor(

            color

        )

    # ---------------------------------------------------------

    def remove_top_right_border(self):

        """
        Cleaner publication style.
        """

        self.axis.spines["top"].set_visible(False)

        self.axis.spines["right"].set_visible(False)

    # ---------------------------------------------------------

    def tight_layout(self):

        self.figure.tight_layout()

    # ---------------------------------------------------------

    @property
    def output_path(self):

        return self.output_dir

    # ---------------------------------------------------------

    @property
    def fig(self):

        return self.figure

    # ---------------------------------------------------------

    @property
    def ax(self):

        return self.axis

    # ---------------------------------------------------------
    # Color Utilities
    # ---------------------------------------------------------

    def set_color_cycle(

        self,

        colors

    ):

        """
        Set custom color cycle.
        """

        self.axis.set_prop_cycle(

            color=colors

        )

    # ---------------------------------------------------------

    def set_background(

        self,

        color="white"

    ):

        """
        Set figure and axis background.
        """

        self.figure.patch.set_facecolor(

            color

        )

        self.axis.set_facecolor(

            color

        )

    # ---------------------------------------------------------
    # Scientific Formatting
    # ---------------------------------------------------------

    def scientific_x(self):

        self.axis.ticklabel_format(

            axis="x",

            style="sci",

            scilimits=(0, 0)

        )

    # ---------------------------------------------------------

    def scientific_y(self):

        self.axis.ticklabel_format(

            axis="y",

            style="sci",

            scilimits=(0, 0)

        )

    # ---------------------------------------------------------
    # Confidence Interval
    # ---------------------------------------------------------

    def confidence_interval(

        self,

        x,

        lower,

        upper,

        alpha=0.25,

        label=None

    ):

        self.axis.fill_between(

            x,

            lower,

            upper,

            alpha=alpha,

            label=label

        )

    # ---------------------------------------------------------
    # Multiple Subplots
    # ---------------------------------------------------------

    def create_subplots(

        self,

        rows,

        cols,

        figsize=None

    ):

        if figsize is None:

            figsize = self.figsize

        self.figure, axes = plt.subplots(

            rows,

            cols,

            figsize=figsize,

            dpi=self.dpi

        )

        return self.figure, axes

    # ---------------------------------------------------------
    # Figure Information
    # ---------------------------------------------------------

    def figure_size(self):

        return self.figure.get_size_inches()

    # ---------------------------------------------------------

    def resolution(self):

        return self.figure.dpi

    # ---------------------------------------------------------
    # Export
    # ---------------------------------------------------------

    def save_png(

        self,

        filename

    ):

        filepath = self.output_dir / filename

        self.tight_layout()

        self.figure.savefig(

            filepath,

            dpi=self.dpi,

            transparent=self.transparent,

            bbox_inches="tight"

        )

        return filepath

    # ---------------------------------------------------------

    def save_pdf(

        self,

        filename

    ):

        filepath = self.output_dir / filename

        self.tight_layout()

        self.figure.savefig(

            filepath,

            format="pdf",

            bbox_inches="tight"

        )

        return filepath

    # ---------------------------------------------------------

    def save_svg(

        self,

        filename

    ):

        filepath = self.output_dir / filename

        self.tight_layout()

        self.figure.savefig(

            filepath,

            format="svg",

            bbox_inches="tight"

        )

        return filepath

    # ---------------------------------------------------------

    def save_jpg(

        self,

        filename,

        quality=95

    ):

        filepath = self.output_dir / filename

        self.tight_layout()

        self.figure.savefig(

            filepath,

            format="jpg",

            dpi=self.dpi,

            pil_kwargs={

                "quality": quality

            }

        )

        return filepath

    # ---------------------------------------------------------
    # Auto Export
    # ---------------------------------------------------------

    def export(

        self,

        filename,

        formats=("png",)

    ):

        exported = []

        stem = Path(filename).stem

        for fmt in formats:

            if fmt == "png":

                exported.append(

                    self.save_png(

                        stem + ".png"

                    )

                )

            elif fmt == "pdf":

                exported.append(

                    self.save_pdf(

                        stem + ".pdf"

                    )

                )

            elif fmt == "svg":

                exported.append(

                    self.save_svg(

                        stem + ".svg"

                    )

                )

            elif fmt in ("jpg", "jpeg"):

                exported.append(

                    self.save_jpg(

                        stem + ".jpg"

                    )

                )

        return exported

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------

    def metadata(

        self

    ):

        return {

            "figure_size":

                tuple(

                    self.figure.get_size_inches()

                ),

            "dpi":

                self.figure.dpi,

            "output_directory":

                str(self.output_dir)

        }

    # ---------------------------------------------------------
    # Reset
    # ---------------------------------------------------------

    def reset(self):

        self.close()

        self.figure = None

        self.axis = None

    # ---------------------------------------------------------
    # Context Manager
    # ---------------------------------------------------------

    def __enter__(self):

        self.create()

        return self

    # ---------------------------------------------------------

    def __exit__(

        self,

        exc_type,

        exc_val,

        exc_tb

    ):

        self.close()

    # ---------------------------------------------------------
    # Representation
    # ---------------------------------------------------------

    def __repr__(self):

        return (

            f"{self.__class__.__name__}"

            f"(output_dir='{self.output_dir}', "

            f"dpi={self.dpi}, "

            f"figsize={self.figsize})"

        )
