from pathlib import Path
dir_path = Path(__file__).parent.absolute()
from .adapter.imagelab import (
    ImagelabIssueFinderAdapter,
    ImagelabDataIssuesAdapter,
    ImagelabReporterAdapter,
)
from .data_issues import DataIssues
from .issue_finder import IssueFinder
from .report import Reporter


def issue_finder_factory(imagelab):
    if imagelab:
        return ImagelabIssueFinderAdapter
    else:
        return IssueFinder


def data_issues_factory(imagelab):
    if imagelab:
        return ImagelabDataIssuesAdapter
    else:
        return DataIssues


def report_factory(imagelab):
    if imagelab:
        return ImagelabReporterAdapter
    else:
        return Reporter
