"""DataLoader: parses the REAL YEDC feeder interruption log
(data/yedc_interruption_log.xlsx, one sheet per month) into a single
clean event-level DataFrame. Corresponds to the DataLoader class in
Chapter 3's Class Diagram (3.2.3.2) and Section 3.4.1.

*** THIS DATA IS REAL, NOT SYNTHETIC. *** It supersedes the earlier
54-event Jan-Feb log: the supervisor supplied an updated workbook
covering January-April 2026 (104 logged interruption events across 3
feeders). Each sheet has a fixed layout: three blank rows, a header row,
then event rows identified by an integer S/N in column 0. Some sheets
also carry a secondary "daily hours interrupted" table in columns 27-37;
that table is only genuinely populated for January and February -- the
March and April copies are stale duplicates of the February table
(verified byte-for-byte identical), so it is NOT used here. Instead,
daily-resolution aggregates are derived directly from the real
event-level records, which do cover the full January-April span.
"""

import pandas as pd

SHEETS = ["JANUARY 2026", "FEBRUARY 2026", "MARCH 2026 ", "APRIL2026"]
N_HEADER_COLS = 27  # columns beyond this are the stale/unused secondary table


class DataLoader:
    def __init__(self, xlsx_path="data/yedc_interruption_log.xlsx"):
        self.xlsx_path = xlsx_path

    def load(self):
        xl = pd.ExcelFile(self.xlsx_path)
        frames = []
        for sheet in SHEETS:
            raw = pd.read_excel(xl, sheet_name=sheet, header=None)
            header = raw.iloc[3, :N_HEADER_COLS].tolist()
            body = raw.iloc[4:, :N_HEADER_COLS].copy()
            body.columns = header
            # Real event rows have an integer S/N; template/blank rows don't.
            is_event = body["S/N"].apply(
                lambda x: isinstance(x, (int, float)) and not pd.isna(x)
            )
            body = body[is_event].copy()
            body["SourceSheet"] = sheet
            frames.append(body)

        events = pd.concat(frames, ignore_index=True)
        events["Affected Feeder"] = events["Affected Feeder"].str.strip()
        events["Date Out"] = pd.to_datetime(events["Date Out"])
        events["Date In"] = pd.to_datetime(events["Date In"])
        events["Duration_Minutes"] = self._parse_duration_minutes(events["Duration"])
        events["Load Loss (MW)"] = pd.to_numeric(events["Load Loss (MW)"], errors="coerce")
        events["Weather Condition"] = events["Weather Condition"].str.strip()
        events = events.sort_values("Date Out").reset_index(drop=True)

        self._events = events
        return events

    @staticmethod
    def _parse_duration_minutes(duration_col):
        """Duration is stored as HH:MM:SS text/time (occasionally with a
        'N day(s), HH:MM:SS' prefix for outages spanning midnight);
        convert to minutes via pandas' Timedelta parser."""
        td = pd.to_timedelta(duration_col.astype(str), errors="coerce")
        return td.dt.total_seconds() / 60

    @property
    def events(self):
        return self._events
