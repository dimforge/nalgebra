/// Examples of *invalid* raw CSR data `(offsets, indices, values)`.
pub struct InvalidCsrDataExamples {
    pub empty_offset_array: (Vec<usize>, Vec<usize>, Vec<i32>),
    pub offset_array_invalid_length_for_arbitrary_data: (Vec<usize>, Vec<usize>, Vec<i32>),
    pub invalid_first_entry_in_offsets_array: (Vec<usize>, Vec<usize>, Vec<i32>),
    pub invalid_last_entry_in_offsets_array: (Vec<usize>, Vec<usize>, Vec<i32>),
    pub invalid_length_of_offsets_array: (Vec<usize>, Vec<usize>, Vec<i32>),
    pub nonmonotonic_offsets: (Vec<usize>, Vec<usize>, Vec<i32>),
    pub nonmonotonic_minor_indices: (Vec<usize>, Vec<usize>, Vec<i32>),
    pub minor_index_out_of_bounds: (Vec<usize>, Vec<usize>, Vec<i32>),
    pub duplicate_entry: (Vec<usize>, Vec<usize>, Vec<i32>),
}

impl InvalidCsrDataExamples {
    pub fn new() -> Self {
        let empty_offset_array = (Vec::<usize>::new(), Vec::<usize>::new(), Vec::<i32>::new());
        let offset_array_invalid_length_for_arbitrary_data =
            (vec![0, 3, 5], vec![0, 1, 2, 3, 5], vec![0, 1, 2, 3, 4]);
        let invalid_first_entry_in_offsets_array =
            (vec![1, 2, 2, 5], vec![0, 5, 1, 2, 3], vec![0, 1, 2, 3, 4]);
        let invalid_last_entry_in_offsets_array =
            (vec![0, 2, 2, 4], vec![0, 5, 1, 2, 3], vec![0, 1, 2, 3, 4]);
        let invalid_length_of_offsets_array =
            (vec![0, 2, 2], vec![0, 5, 1, 2, 3], vec![0, 1, 2, 3, 4]);
        let nonmonotonic_offsets = (vec![0, 3, 2, 5], vec![0, 1, 2, 3, 4], vec![0, 1, 2, 3, 4]);
        let nonmonotonic_minor_indices =
            (vec![0, 2, 2, 5], vec![0, 2, 3, 1, 4], vec![0, 1, 2, 3, 4]);
        let minor_index_out_of_bounds =
            (vec![0, 2, 2, 5], vec![0, 6, 1, 2, 3], vec![0, 1, 2, 3, 4]);
        let duplicate_entry = (vec![0, 2, 2, 5], vec![0, 5, 2, 2, 3], vec![0, 1, 2, 3, 4]);

        return Self {
            empty_offset_array,
            offset_array_invalid_length_for_arbitrary_data,
            invalid_first_entry_in_offsets_array,
            invalid_last_entry_in_offsets_array,
            invalid_length_of_offsets_array,
            nonmonotonic_minor_indices,
            nonmonotonic_offsets,
            minor_index_out_of_bounds,
            duplicate_entry,
        };
    }
}
