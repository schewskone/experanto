import numpy as np

from experanto.interpolators import SequenceInterpolator
from experanto.intervals import (
    TimeInterval,
    find_complement_of_interval_array,
    uniquefy_interval_array,
)


def nan_filter(vicinity=0.05):
    """
    Create a filter that identifies valid time intervals excluding regions near NaN values.

    This filter detects NaN values in sequence data and marks surrounding time intervals
    as invalid to avoid using corrupted or unreliable data near gaps.

    Args:
        vicinity (float, optional): Time window in seconds around each NaN to mark as invalid.
            For example, vicinity=0.05 marks 50ms before and after each NaN as invalid.
            Defaults to 0.05.

    Returns:
        callable: A filter function that takes a SequenceInterpolator and returns a list
            of valid TimeInterval objects.

    Example:
        >>> from experanto.interpolators import SequenceInterpolator
        >>> filter_fn = nan_filter(vicinity=0.1)  # 100ms around NaNs
        >>> interpolator = SequenceInterpolator("path/to/data")
        >>> valid_intervals = filter_fn(interpolator)
        >>> # Use valid_intervals to exclude problematic time regions

    Note:
        This filter requires a SequenceInterpolator because it uses time_delta internally.
        Other interpolator types don't have this attribute.
    """

    def implementation(device_: SequenceInterpolator):
        # requests SequenceInterpolator as uses time_delta internally
        # and other interpolators don't have it
        time_delta = device_.time_delta
        start_time = device_.start_time
        end_time = device_.end_time
        data = device_._data  # (T, n_neurons)

        # detect nans
        nan_mask = np.isnan(data)  # (T, n_neurons)
        nan_mask = np.any(nan_mask, axis=1)  # (T,)

        # Find indices where nan_mask is True
        nan_indices = np.where(nan_mask)[0]

        # Create invalid TimeIntervals around each nan point
        invalid_intervals = []
        vicinity_seconds = vicinity  # vicinity is already in seconds
        for idx in nan_indices:
            time_point = start_time + idx * time_delta
            interval_start = max(start_time, time_point - vicinity_seconds)
            interval_end = min(end_time, time_point + vicinity_seconds)
            invalid_intervals.append(TimeInterval(interval_start, interval_end))

        # Merge overlapping invalid intervals
        invalid_intervals = uniquefy_interval_array(invalid_intervals)

        # Find the complement of invalid intervals to get valid intervals
        valid_intervals = find_complement_of_interval_array(
            start_time, end_time, invalid_intervals
        )

        return valid_intervals

    return implementation
