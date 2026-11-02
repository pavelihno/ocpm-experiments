from abc import ABC, abstractmethod

import networkx as nx
import pandas as pd

from utils.constants import (
    EXECUTION_ID,
    OCEL_EID,
    OCEL_OID,
    OCEL_TIMESTAMP,
    PREFIX_ID,
    PREFIX_LEN,
    REMAINING_TIME,
)
from utils.preprocess import create_unbiased_ocel_split


class BaseOCELDataset(ABC):
    def __init__(
        self,
        ocel,
        train_ratio=0.8,
        min_prefix=1,
        max_prefix=None,
    ):
        """Base dataset class for Object-Centric Event Logs."""

        self.ocel = ocel
        self.train_ratio = train_ratio
        self.min_prefix = min_prefix
        self.max_prefix = max_prefix

        # Build isolated process executions from the graph
        self.raw_df = self._extract_executions()

        # Enforce types and sorting for temporal consistency
        self.raw_df[EXECUTION_ID] = self.raw_df[EXECUTION_ID].astype(str)
        self.raw_df[OCEL_TIMESTAMP] = pd.to_datetime(
            self.raw_df[OCEL_TIMESTAMP], utc=True
        )
        self.raw_df = self.raw_df.sort_values(
            [OCEL_TIMESTAMP, EXECUTION_ID]
        ).reset_index(drop=True)

    def _extract_executions(self):
        """
        Builds a bipartite graph of events and objects to find connected components.
        Each connected component represents an isolated process execution.
        """
        events_df = self.ocel.events.copy()
        relations_df = self.ocel.relations

        G = nx.Graph()

        # Add edges between events and objects (prefixing to avoid ID collision)
        for _, row in relations_df.iterrows():
            G.add_edge(f'e_{row[OCEL_EID]}', f'o_{row[OCEL_OID]}')

        components = list(nx.connected_components(G))
        execution_mapping = []

        for exec_idx, comp in enumerate(components):
            # Extract only the event nodes for this specific component
            event_ids = [node[2:] for node in comp if node.startswith('e_')]

            for eid in event_ids:
                execution_mapping.append(
                    {OCEL_EID: eid, EXECUTION_ID: f'exec_{exec_idx}'}
                )

        mapping_df = pd.DataFrame(execution_mapping)

        if not mapping_df.empty:
            # Merge execution IDs back into the main events DataFrame
            events_df = events_df.merge(mapping_df, on=OCEL_EID, how='inner')
        else:
            events_df[EXECUTION_ID] = []

        return events_df

    @abstractmethod
    def get_prefixes_and_labels(self, df):
        """Generate all prefixes and extract corresponding KPI labels."""
        pass

    def train_test_split(self):
        """
        Create train/test split with temporal debiasing (Weytjens methodology).
        """
        if self.raw_df is None or self.raw_df.empty:
            raise ValueError('Dataset not loaded or is empty.')

        train_df, test_df = create_unbiased_ocel_split(
            self.raw_df,
            time_col=OCEL_TIMESTAMP,
            case_col=EXECUTION_ID,
            test_share=(1 - self.train_ratio),
        )

        return train_df, test_df


class RemainingTimeDataset(BaseOCELDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_prefixes_and_labels(self, df):
        """
        Generate prefixes and extract the remaining time for each prefix state.
        """
        prefixes = []
        labels = []

        for exec_id, group in df.groupby(EXECUTION_ID):
            events = group.to_dict('records')
            limit = (
                min(len(events), self.max_prefix) if self.max_prefix else len(events)
            )

            for i in range(self.min_prefix, limit + 1):
                prefix_slice = [dict(e) for e in events[:i]]

                # Tag the prefix slice with metadata
                for e in prefix_slice:
                    e[PREFIX_ID] = f'{exec_id}_prefix_{i}'
                    e[PREFIX_LEN] = i

                prefixes.extend(prefix_slice)

                current_event = prefix_slice[-1]
                labels.append(
                    {
                        PREFIX_ID: f'{exec_id}_prefix_{i}',
                        REMAINING_TIME: current_event.get(REMAINING_TIME, 0.0),
                    }
                )

        prefixes_df = pd.DataFrame(prefixes)
        labels_df = pd.DataFrame(labels).set_index(PREFIX_ID)

        return prefixes_df, labels_df
