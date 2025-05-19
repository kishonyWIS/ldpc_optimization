import numpy as np
import random


class StabilizerCode:
    def __init__(self, x_stabilizers, z_stabilizers, lx, lz):
        """
        Initialize a StabilizerCode instance.

        Parameters:
          x_stabilizers : list of list of int
              Each sublist represents an X stabilizer (its support: a list of data qubit indices).
          z_stabilizers : list of list of int
              Each sublist represents a Z stabilizer (its support).
          lx : np.ndarray
              The logical operator matrix for X logicals (each row corresponds to one logical observable).
          lz : np.ndarray
              The logical operator matrix for Z logicals.

        The total number of data qubits, n, is computed as one plus the maximum data qubit index.
        """
        self.x_stabilizers = x_stabilizers
        self.z_stabilizers = z_stabilizers
        self.lx = lx
        self.lz = lz

        all_qubits = {q for stab in x_stabilizers for q in stab}
        all_qubits |= {q for stab in z_stabilizers for q in stab}
        self.n = max(all_qubits) + 1

    def generate_cx_list(self):
        """
        Generate a CX list ordering for the code, where each stabilizer is measured sequentially.

        Returns:
          cx_list : list of tuple
              Each tuple (q, a) represents a CX gate with q a data qubit index and a an ancilla identifier.
              X stabilizers are processed first, followed by Z stabilizers.
        """
        cx_list = []
        for ancilla, qubits in zip(self.get_x_ancillas(), self.x_stabilizers):

            for q in qubits:
                cx_list.append((q, ancilla))
        for ancilla, qubits in zip(self.get_z_ancillas(), self.z_stabilizers):
            for q in qubits:
                cx_list.append((q, ancilla))
        return cx_list

    def build_mappings(self):
        """
        Build the dictionaries required for circuit construction.

        Returns:
          ancilla_type : dict
              Maps each ancilla identifier (e.g. "aX0", "aZ0") to its type ("X" or "Z").
          data_mapping : dict
              Maps each data qubit index (0 to n-1) to its physical index (here, the identity).
          ancilla_mapping : dict
              Assigns a physical index to each ancilla (starting after the data qubits).
        """
        ancilla_type = {}
        for a in self.get_x_ancillas():
            ancilla_type[a] = "X"
        for a in self.get_z_ancillas():
            ancilla_type[a] = "Z"
        data_mapping = {q: q for q in range(self.n)}
        ancilla_mapping = {}
        flag_mapping = {}
        next_index = self.n
        for ancilla in sorted(ancilla_type.keys()):
            ancilla_mapping[ancilla] = next_index
            next_index += 1

        for ancilla in sorted(ancilla_type.keys()):
            flag_mapping[ancilla] = next_index
            next_index += 1

        return ancilla_type, data_mapping, ancilla_mapping, flag_mapping

    def get_x_ancillas(self):
        """Return a list of ancilla IDs for X stabilizers."""
        return [f"X{i}" for i in range(len(self.x_stabilizers))]

    def get_z_ancillas(self):
        """Return a list of ancilla IDs for Z stabilizers."""
        return [f"Z{i}" for i in range(len(self.z_stabilizers))]

    def __str__(self):
        s = f"StabilizerCode(n={self.n},\n"
        s += f"  x_stabilizers={self.x_stabilizers},\n"
        s += f"  z_stabilizers={self.z_stabilizers},\n"
        s += f"  lx=\n{self.lx},\n"
        s += f"  lz=\n{self.lz})"
        return s


def generate_optimally_ordered_rotated_surface_code_stabilizers(L):
    """
    Generate the stabilizers for a rotated surface code on an L×L lattice (L odd).

    Data qubits are arranged in L rows (row 0 to L-1) and L columns (col 0 to L-1) in row‐major order.
    In the interior, every 2×2 block (with top‐left corner at (i,j) for i,j=0,...,L-2) is used,
    with type determined by (i+j) mod 2 (if even, assign to X; if odd, assign to Z).

    Returns:
      x_stabilizers, z_stabilizers : two lists of lists of int.
    """
    x_stab = []
    z_stab = []
    # Interior 2x2 blocks.
    for i in range(L - 1):
        for j in range(L - 1):
            block = [i * L + j, i * L + j + 1,
                     (i + 1) * L + j, (i + 1) * L + j + 1]
            if (i + j) % 2 == 0:
                x_stab.append(block)
            else:
                z_stab.append(block)
    # Reorder each list inside x_stab as: 2nd biggest, smallest, biggest, 2nd smallest.
    optimally_ordered_x_stab = []
    optimally_ordered_z_stab = []
    for stab in x_stab:
        optimally_ordered_x_stab.append(
            [stab[2], stab[0], stab[3], stab[1]])
    for stab in z_stab:
        optimally_ordered_z_stab.append(
            [stab[1], stab[0], stab[3], stab[2]])

    # Top boundary: row 0.
    top = list(range(L))
    # Partition top row into (L-1)//2 pairs using even-indexed pairs.
    for k in range((L - 1) // 2):
        pair = [top[2 * k+1], top[2 * k]]

        # Top boundary stabilizers are Z.
        optimally_ordered_z_stab.append(pair)
    # Bottom boundary: row L-1.
    bottom = list(range((L - 1) * L, L * L))
    for k in range((L - 1) // 2):
        # For bottom, choose the last pair.
        pair = [bottom[-1 - 2 * k], bottom[-2 - 2 * k]]
        optimally_ordered_z_stab.append(pair)
    # Left boundary: column 0.
    left = [i * L for i in range(L)]
    for k in range(1, (L - 1) // 2 + 1):
        pair = [left[2 * k], left[2 * k-1]]
        optimally_ordered_x_stab.append(pair)
    # Right boundary: column L-1.
    right = [i * L + (L - 1) for i in range(L)]
    for k in range(1, (L - 1) // 2 + 1):
        pair = [right[2 * k - 2], right[2 * k - 1]]
        optimally_ordered_x_stab.append(pair)

    return optimally_ordered_x_stab, optimally_ordered_z_stab


def generate_random_ordered_rotated_surface_code_stabilizers(L):
    """
    Generate the stabilizers for a rotated surface code on an L×L lattice (L odd).

    Data qubits are arranged in L rows (row 0 to L-1) and L columns (col 0 to L-1) in row‐major order.
    In the interior, every 2×2 block (with top‐left corner at (i,j) for i,j=0,...,L-2) is used,
    with type determined by (i+j) mod 2 (if even, assign to X; if odd, assign to Z).

    On the boundaries, we add additional two‑body stabilizers:
      - Top boundary (row 0): partition the top row into (L-1)/2 pairs.
      - Bottom boundary (row L-1): partition the row into (L-1)/2 pairs.
      - Left boundary (column 0): use pairs from the left column (excluding the top element).
      - Right boundary (column L-1): use pairs from the right column (excluding the bottom element).

    Returns:
      x_stabilizers, z_stabilizers : two lists of lists of int.
    """
    x_stab = []
    z_stab = []
    # Interior 2x2 blocks.
    for i in range(L - 1):
        for j in range(L - 1):
            block = [i * L + j, i * L + j + 1,
                     (i + 1) * L + j, (i + 1) * L + j + 1]
            if (i + j) % 2 == 0:
                x_stab.append(block)
            else:
                z_stab.append(block)
    # Top boundary: row 0.
    top = list(range(L))
    # Partition top row into (L-1)//2 pairs using even-indexed pairs.
    for k in range((L - 1) // 2):
        pair = [top[2 * k], top[2 * k + 1]]
        z_stab.append(pair)  # Top boundary stabilizers are Z.
    # Bottom boundary: row L-1.
    bottom = list(range((L - 1) * L, L * L))
    for k in range((L - 1) // 2):
        # For bottom, choose the last pair.
        pair = [bottom[-2 - 2 * k], bottom[-1 - 2 * k]]
        z_stab.append(pair)
    # Left boundary: column 0.
    left = [i * L for i in range(L)]
    for k in range(1, (L - 1) // 2 + 1):
        pair = [left[2 * k - 1], left[2 * k]]
        x_stab.append(pair)
    # Right boundary: column L-1.
    right = [i * L + (L - 1) for i in range(L)]
    for k in range(1, (L - 1) // 2 + 1):
        pair = [right[2 * k - 2], right[2 * k - 1]]
        x_stab.append(pair)

    # Randomize the order of qubits in each stabilizer.
    for stab in x_stab:
        random.shuffle(stab)
    for stab in z_stab:
        random.shuffle(stab)
    return x_stab, z_stab


class RotatedSurfaceCode(StabilizerCode):
    def __init__(self, L, ordering: str = 'random'):
        """
        Initialize a RotatedSurfaceCode on an L×L lattice (L odd) with given logical operators lx and lz.

        Parameters:
          L : int
              The lattice dimension (number of data qubits per row); must be odd and at least 3.
          lx : np.ndarray
              The logical operator matrix for X logicals.
          lz : np.ndarray
              The logical operator matrix for Z logicals.
        """
        n = L * L
        # Define trivial logical operators.
        # For example, logical X along the first row (indices 0 to L-1).
        lx = np.zeros((1, n), dtype=int)
        lx[0, :L] = 1
        # Logical Z along the first column (indices 0, L, 2L, ...).
        lz = np.zeros((1, n), dtype=int)
        for ii in range(L):
            lz[0, ii * L] = 1

        if L < 3 or L % 2 == 0:
            raise ValueError("L must be an odd integer >= 3.")
        if ordering == 'random':
            x_stab, z_stab = generate_random_ordered_rotated_surface_code_stabilizers(
                L)
        elif ordering == 'optimal':
            x_stab, z_stab = generate_optimally_ordered_rotated_surface_code_stabilizers(
                L)
        else:
            raise ValueError("Invalid ordering. Use 'random' or 'optimal'.")
        super().__init__(x_stab, z_stab, lx, lz)
        self.L = L

        # Automatically assign coordinates.
        self.data_coords, self.ancilla_coords = self.assign_coordinates()

    def assign_coordinates(self):
        """
        Compute coordinates for all qubits (data and ancilla).

        Data qubits are arranged in an L×L grid in row-major order.
        We assign coordinates as (col, L-1-row) so that the origin (0,0) is at the bottom left.

        For ancillas, the coordinate is defined as the average of the coordinates of the data qubits
        in the corresponding stabilizer.

        Returns:
          data_coords : dict
              Maps data qubit index to (x, y) coordinate.
          ancilla_coords : dict
              Maps ancilla ID to (x, y) coordinate.
        """
        data_coords = {}
        L = self.L
        # Data qubits: q = i * L + j has coordinate (j, L - 1 - i).
        for i in range(L):
            for j in range(L):
                q = i * L + j
                data_coords[q] = (j, L - 1 - i)

        ancilla_coords = {}
        # For each X stabilizer ancilla:
        for idx, support in enumerate(self.x_stabilizers):
            ancilla_id = f"X{idx}"
            coords = [data_coords[q] for q in support]
            # Compute average coordinate.
            avg_x = np.mean([c[0] for c in coords])
            avg_y = np.mean([c[1] for c in coords])
            ancilla_coords[ancilla_id] = (avg_x, avg_y)
        # For each Z stabilizer ancilla:
        for idx, support in enumerate(self.z_stabilizers):
            ancilla_id = f"Z{idx}"
            coords = [data_coords[q] for q in support]
            avg_x = np.mean([c[0] for c in coords])
            avg_y = np.mean([c[1] for c in coords])
            ancilla_coords[ancilla_id] = (avg_x, avg_y)
        return data_coords, ancilla_coords

    def __str__(self):
        s = f"RotatedSurfaceCode(L={self.L}, n={self.n})\n"
        s += f"X stabilizers ({len(self.x_stabilizers)}): {self.x_stabilizers}\n"
        s += f"Z stabilizers ({len(self.z_stabilizers)}): {self.z_stabilizers}\n"
        s += f"Logical X operator (lx):\n{self.lx}\n"
        s += f"Logical Z operator (lz):\n{self.lz}\n"
        return s


# --- Example usage ---
if __name__ == "__main__":
    # For a rotated surface code with L = 3 (3x3 lattice).
    L = 3

    code = RotatedSurfaceCode(L)
    print("Rotated Surface Code Definition:")
    print(code)

    cx_list = code.generate_cx_list()
    print("Generated CX List Ordering:")
    for i, gate in enumerate(cx_list):
        print(f"{i}: {gate}")

    ancilla_type, data_mapping, ancilla_mapping, flag_mapping = code.build_mappings()
    print("Ancilla Type Dictionary:")
    print(ancilla_type)
    print("Data Mapping:")
    print(data_mapping)
    print("Ancilla Mapping:")
    print(ancilla_mapping)
