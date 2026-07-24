"""
=========================================================
SleepTransitionFramework

ANPHY Dataset

Dataset interface for the ANPHY Sleep Database.

Dataset Structure
-----------------

ANPHY/

    Subjects/

        EPCTL01/

            EPCTL01.edf

            EPCTL01-annotation.txt

            EPCTL01_artndxn.mat

        ...

    Details information of subjects.xlsx

Author
------

Pushkar Singh
=========================================================
"""

from __future__ import annotations

from pathlib import Path

from src.loaders.edf_loader import EDFLoader
from src.loaders.annotation_loader import AnnotationLoader
from src.loaders.artifact_loader import ArtifactLoader
from src.loaders.subject_info_loader import SubjectInfoLoader


class ANPHYDataset:
    """
    ANPHY Sleep Dataset Interface.
    """

    # -----------------------------------------------------

    def __init__(

        self,

        root

    ):

        self.root = Path(root)

        self.subject_root = self.root / "Subjects"

        self.subject_info = (

            self.root /

            "Details information of subjects.xlsx"

        )

        self.edf_loader = EDFLoader()

        self.annotation_loader = AnnotationLoader()

        self.artifact_loader = ArtifactLoader()

        self.subject_loader = SubjectInfoLoader()

    # -----------------------------------------------------

    def subjects(self):
        """
        Return all subject folders.
        """

        return sorted(

            [

                folder

                for folder in self.subject_root.iterdir()

                if folder.is_dir()

            ]

        )

    # -----------------------------------------------------

    def number_of_subjects(self):

        return len(

            self.subjects()

        )

    # -----------------------------------------------------

    def subject_ids(self):

        return [

            folder.name

            for folder in self.subjects()

        ]

    # -----------------------------------------------------

    def subject_folder(

        self,

        subject

    ):

        return self.subject_root / subject

    # -----------------------------------------------------

    def edf_file(

        self,

        subject

    ):

        folder = self.subject_folder(subject)

        return folder / f"{subject}.edf"

    # -----------------------------------------------------

    def annotation_file(

        self,

        subject

    ):

        folder = self.subject_folder(subject)

        return folder / f"{subject}-annotation.txt"

    # -----------------------------------------------------

    def artifact_file(

        self,

        subject

    ):

        folder = self.subject_folder(subject)

        return folder / f"{subject}_artndxn.mat"

    # -----------------------------------------------------

    def load_edf(

        self,

        subject

    ):

        return self.edf_loader.load(

            self.edf_file(subject)

        )

    # -----------------------------------------------------

    def load_annotations(

        self,

        subject

    ):

        return self.annotation_loader.load(

            self.annotation_file(subject)

        )

    # -----------------------------------------------------

    def load_artifacts(

        self,

        subject

    ):

        return self.artifact_loader.load(

            self.artifact_file(subject)

        )

    # -----------------------------------------------------

    def load_subject_information(self):

        return self.subject_loader.load(

            self.subject_info

        )

    # -----------------------------------------------------

    def verify_subject(

        self,

        subject

    ):

        return {

            "edf":

                self.edf_file(subject).exists(),

            "annotation":

                self.annotation_file(subject).exists(),

            "artifact":

                self.artifact_file(subject).exists()

        }

    # -----------------------------------------------------

    def summary(self):

        print()

        print("=" * 70)

        print("ANPHY DATASET")

        print("=" * 70)

        print()

        print(

            "Root :", self.root

        )

        print()

        print(

            "Subjects :",

            self.number_of_subjects()

        )

        print()

        print("=" * 70)

    # -----------------------------------------------------

    def __len__(self):

        return self.number_of_subjects()

    # -----------------------------------------------------

    def __repr__(self):

        return (

            f"ANPHYDataset("

            f"subjects={len(self)})"

        )
