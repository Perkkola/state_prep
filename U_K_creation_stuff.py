def generate_cnots(arr):
        temp = []
        arr = np.array(arr).transpose().tolist()
        for i in range(2):
            for a in arr:
                cp = np.copy(a).tolist()
                cp.append(i)
                temp.append(cp)

        return np.array(temp).transpose().tolist()

def generate_thetas(arr):
    arr = np.array(arr) / 2
    arr = np.append(arr, arr * -1)
    return arr.tolist()

def flip_signs(cnots, thetas, x_index_list):
    arr = np.copy(cnots).transpose().tolist()
    thetas = np.copy(thetas)
    for index, bit_list in enumerate(arr):
        xor_list = [bit_list[x] for x in x_index_list]
        xor_result = reduce(lambda a, b: a ^ b, xor_list)
        if xor_result == 1: thetas[index] *= -1
    return thetas


def create_phase_x_index_list(qubit_length):
    x_index_list = []

    for i in range(1, 2 ** qubit_length):
        binary_string = ("{:0{width}b}".format(i, width=qubit_length))[::-1]
        indices = [i + 1 for i, x in enumerate(binary_string) if x == '1']
        x_index_list.append(indices)
    print(x_index_list)
    exit()
    return x_index_list

def createU_k(circuit, data_arr):
        cnots = [[1]]
        all_thetas = []
        for _ in range(QPU_len):
            cnots = generate_cnots(cnots)

        for theta in data_arr:
            thetas = [2 * theta]
            for _ in range(QPU_len):
                thetas = generate_thetas(thetas)
            all_thetas.append(thetas)


        thetas = np.copy(all_thetas[-1])
        x_phase_index_lists = create_phase_x_index_list(QPU_len)[::-1]
        
        for thetas_cp, x_phase_index_list in zip(all_thetas[:-1], x_phase_index_lists):
            flipped_thetas = flip_signs(cnots, np.copy(thetas_cp), x_phase_index_list)
            thetas = np.array(thetas) + np.array(flipped_thetas)

        thetas = thetas.tolist()

        gray_qc = synth_cnot_phase_aam(cnots, thetas)

        circuit.append(gray_qc.to_gate(), [x for x in range(QPU_len + 1)])