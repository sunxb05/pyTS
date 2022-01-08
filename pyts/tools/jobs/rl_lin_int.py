import sys
import numpy as np

class MOL:
    """The MOL class allow the loading and the manipulation of standard XMOL formatted cartesian coordinates"""
    def __init__(self, path):
        print("\nXYZ Loading...\t\t\t", end="", flush=True)
        with open(path) as file:
            lines = file.readlines()
        self.natoms = int(lines[0])
        self.element = []
        self.xyz = []
        for line in lines[2:]:
            line = line.split()
            self.element.append(line[0])
            self.xyz.append([float(line[1]), float(line[2]), float(line[3])])

class IntMOL:

    def __init__(self, path):

        mol_A = MOL(sys.argv[1])
        mol_B = MOL(sys.argv[2])

        nint = 500
        trj = []
        diff = []
        for atom_A, atom_B in zip(mol_A.xyz, mol_B.xyz):
            diff.append([
                (atom_A[0] - atom_B[0]) / (nint + 1),
                (atom_A[1] - atom_B[1]) / (nint + 1),
                (atom_A[2] - atom_B[2]) / (nint + 1)
            ])

        for frame in range(1, nint + 1):
            mol_frame = []
            for atom, ato_diff in zip(mol_A.xyz, diff):
                mol_frame.append([
                    atom[0] - ato_diff[0] * frame,
                    atom[1] - ato_diff[1] * frame,
                    atom[2] - ato_diff[2] * frame
                ])
            trj.append(mol_frame)

        monitor = [int(i) for i in input("\nCoordinates to monitor:\t").split()]
        mon_trj = []
        if len(monitor) == 2:
            print("\nDIST A:\t{}\nDIST B:\t{}".format(
                self.dist(mol_A.xyz[monitor[0] - 1], mol_A.xyz[monitor[1] - 1]),
                self.dist(mol_B.xyz[monitor[0] - 1], mol_B.xyz[monitor[1] - 1])
            ))
            for frame in trj:
                mon_trj.append(self.dist(frame[monitor[0] - 1], frame[monitor[1] - 1]))


        mon_tgt = float(input("\nTARGET:\t"))
        closest_frame = min(range(len(mon_trj)), key=lambda i: abs(mon_trj[i] - mon_tgt))

        print("\nThe closest frame is the number: {}".format(closest_frame))
        self.write_xmol(mol_A.element, trj[closest_frame], "./inter.xyz")
        print("- Interpolated structure has been saved as interp.xyz")
        self.write_traj(mol_A.element, trj, "./trj.xyz")
        print("- Interpolated trajectory has been saved as trj.xyz")

    def write_xmol(elements, coordinates, path):
            """Saving to the "path" a XMol type file from a list of "elements" and their
            "coordinates" arranged as a list of lists ordered as [[x1, y1, z1],[x2, y2, z2]...]"""
            draft = open(path, "w")
            draft.write("{}\n\n".format(str(len(elements))))
            for idnx, val in enumerate(elements):
                draft.write("{} \t{:.10f}\t {:.10f}\t {:.10f} \n".format(
                    val,
                    coordinates[idnx][0],
                    coordinates[idnx][1],
                    coordinates[idnx][2]
                    ))

    def write_traj(elements, coordinateset, path):

            draft = open(path, "w")
            for frame in coordinateset:
                draft.write("{}\n\n".format(str(len(elements))))
                for idnx, val in enumerate(elements):
                    draft.write("{} \t{:.10f}\t {:.10f}\t {:.10f} \n".format(
                        val,
                        frame[idnx][0],
                        frame[idnx][1],
                        frame[idnx][2]
                        ))
                draft.write("\n")

    def dist(atom_a, atom_b):
        """Calulate the distance between two atoms

        Args:
            atom_a (list): Cartesian coordinate of atom a [X, Y, Z]
            atom_b (list): Cartesian coordinate of atom b [X, Y, Z]
        Returns:
            [type]: Distance between atom_a and atom_b
        """
        atom_a = np.array(atom_a)
        atom_b = np.array(atom_b)
        return np.linalg.norm(atom_a - atom_b)
