import os
import pickle
from braket.aws import AwsDevice

from braket.circuits import Circuit

def append_new_line(file_name, text_to_append):
    """
    Helper function to store job ids.
    Append given text as a new line at the end of file
    """
    # Open the file in append & read mode ('a+')
    with open(file_name, "a+") as file_object:
        # Move read cursor to the start of file.
        file_object.seek(0)
        # If file is not empty then append '\n'
        data = file_object.read(100)
        if len(data) > 0:
            file_object.write("\n")
        # Append text at the end of file
        file_object.write(text_to_append)

#-----------------------
# PART 2: Translate circuits to IBM / QISKIT FORMAT
#-----------------------
def _circuits_translator(eigenstate_index: int,
                            quantum_circuit: Circuit,
                            qubit_index: int):
    """
    Helper method that creates both DDOT and QDOT circuits in format native to BRAKET and applies to existing (empty) braket_circuit,
     by applying Pauli eigenstates (formerly _apply_pauli_eigenstate).
    """
    # _pauli_labels = ['z+', 'z-', 'x+', 'x-', 'y+', 'y-']

    # Z+
    if eigenstate_index == 0:
        quantum_circuit = quantum_circuit.rz(qubit_index, 0)

    # Z-
    elif eigenstate_index == 1:
        # quantum_circuit = quantum_circuit.rx(qubit_index, np.pi)
        quantum_circuit = quantum_circuit.rz(qubit_index, -0.9199372448290238)
        quantum_circuit = quantum_circuit.rx(qubit_index, np.pi / 2)
        quantum_circuit = quantum_circuit.rz(qubit_index, np.pi)
        quantum_circuit = quantum_circuit.rx(qubit_index, - np.pi / 2)
        quantum_circuit = quantum_circuit.rz(qubit_index, 2.2216554087607694)

    # X+
    elif eigenstate_index == 2:
        quantum_circuit = quantum_circuit.rz(qubit_index, np.pi / 2)
        quantum_circuit = quantum_circuit.rx(qubit_index, np.pi / 2)
        quantum_circuit = quantum_circuit.rz(qubit_index, np.pi / 2)

    # X-
    elif eigenstate_index == 3:
        quantum_circuit = quantum_circuit.rz(qubit_index, np.pi / 2)
        quantum_circuit = quantum_circuit.rx(qubit_index, np.pi / 2)
        quantum_circuit = quantum_circuit.rz(qubit_index, -np.pi / 2)

    # Y+
    elif eigenstate_index == 4:
        quantum_circuit = quantum_circuit.rz(qubit_index, np.pi / 2)
        quantum_circuit = quantum_circuit.rx(qubit_index, np.pi / 2)
        quantum_circuit = quantum_circuit.rz(qubit_index, np.pi)

    # Y-
    elif eigenstate_index == 5:
        quantum_circuit = quantum_circuit.rz(qubit_index, np.pi / 2)
        quantum_circuit = quantum_circuit.rx(qubit_index, np.pi / 2)

    else:
        raise ValueError(f"Incorrect eigenstate index: '{eigenstate_index}'!")

    return quantum_circuit

def run():
    """
    Entrypoint function to run prepared experiment.
    See https://amazon-braket-sdk-python.readthedocs.io/en/stable/_apidoc/braket.aws.aws_quantum_job.html
    Write in create  AwsQuantumJob: entry_point="aws_braket_runner:run"
    Should run until all job/tasks are completed
    """
    #[1]Entry definityions
    device = AwsDevice(os.environ["AMZN_BRAKET_DEVICE_ARN"])
    input_dir = os.environ["AMZN_BRAKET_INPUT_DIR"]
    output_dir = os.environ["AMZN_BRAKET_JOB_RESULTS_DIR"]

    circuits_file = f'{input_dir}/braket_circuits_list/braket_circuits_list.pkl'
    circuits_symbolic_file =  f'{input_dir}/qrem_circuits_list/qrem_circuits_list.txt'
    metadata_file = f'{input_dir}/metadata/metadata.pkl'
    good_qubits_indices_file = f'{input_dir}/good_qubits_indices/good_qubits_indices.txt'
    task_arns_file = f'{output_dir}/task_arns.txt'


    #[0] Prepare valid qubit indices list
    #------------------------------------
    good_qubits_indices = []
    with open(good_qubits_indices_file, 'r') as fp:
        for line in fp:
            # remove linebreak from a current name
            # linebreak is the last character of each line
            x = line[:-1]
            good_qubits_indices.append(int(x))
    
    #[1] Load pickled braket circuits and metadata
    #---------------------------------------------
    braket_circuits =[]
    circuits_symbolic = []
    results = []  #ADD
    task_arn_list = []  #ADD
    metadata = {}
    pickle_load = False

    with open(metadata_file, 'rb') as fm:
        metadata = pickle.load(fm) # deserialize using load()
        

    with open(circuits_symbolic_file, 'r') as fp:
        for line in fp:
            # remove linebreak from a current name
            # linebreak is the last character of each line
            x = line[:-1]
            new_circuit = []
            for label in x:
                new_circuit.append(int(label))
            circuits_symbolic.append(new_circuit)


    if os.path.isfile(circuits_file):
        with open(circuits_file, 'rb') as f:
            braket_circuits = pickle.load(f) # deserialize using load()
    #[1.1] If pickled circuits were not provided:
    #---------------------------------------------
    else:
        circuits_number = len(circuits_symbolic)
        for circuit_idx in range(circuits_number):
            aws_circuit = Circuit()
            for idx, label in enumerate(circuits_symbolic[circuit_idx]):
                aws_circuit = _circuits_translator(eigenstate_index=label, quantum_circuit=aws_circuit,
                                                    qubit_index=good_qubits_indices[idx])
            aws_circuit = Circuit().add_verbatim_box(aws_circuit)
            braket_circuits.append(aws_circuit)


    # [2] prepare tags if present in metadata
    #------------------------------------------------------------------------------------------------------------
    number_of_shots = metadata["number_of_repetitions"]
    MAX_RETRIES = metadata.get("max_task_retries", 3)  
    tags = metadata.get("tags", {}) 
    
    # [3] run tasks consecutively. According to aws, this ensures order of exxecution. We also save circuit number as a hash
    #------------------------------------------------------------------------------------------------------------
    #TODO czy circuit to jest reprezentacja binarna czy obiekt braketu?
    tasks = [device.run(circuit, 
                        shots=number_of_shots, 
                        disable_qubit_rewiring=True, 
                        tags={**tags, "circuit_id": idc, "circuit": circuits_symbolic[idc]}) 
                        for (idc, circuit) in enumerate(braket_circuits)]
     
    # now tasks will be saved on s3. However, we can aggregate all results in a single results file in output folder - point [5] below
    # [4] retry failed tasks
    #------------------------------------------------------------------------------------------------------------
    
    retries = 0

    results = [task.result() for task in tasks]
    while retries < MAX_RETRIES and not all(results): 
        for i, result in enumerate(results):
            if not result:
                print(f"Task {tasks[i].arn} had failed. Retrying")
                tasks[i] =device.run(braket_circuits[i], 
                                    shots=number_of_shots, 
                                    disable_qubit_rewiring=True, 
                                    tags={**tags, "circuit_id": i, "circuit": circuits_symbolic[i]}) 
        results = [task.result() for task in tasks]
        retries += 1      

    failed_tasks = [(i, task.arn) for (i, task) in enumerate(tasks) if not results[i]]
    
    # [5] Save job results and arns of failed tasks
    #------------------------------------------------------------------------------------------------------------
    
    task_arn_list = [task.arn for task in tasks]
    
    #TODO unikalność circuits - czy jest unikalne?
    measurements = {str(circuit):result.measurement_counts if result else None for (circuit,result) in zip(circuits_symbolic,results) }
    save_job_result({ "measurement_counts": measurements, 
                      "task_arn_list": task_arn_list, 
                      "failed_tasks": failed_tasks})
    

    #[5.1] just as a backup save all arns for the given job
    #------------------------------------------------------
    for task in tasks:
        append_new_line(task_arns_file, task.id)
    
    
    
    # OLD VERSION BELOW
    
    # #  divide the circuits into 1-circuit batches (we use run_batches() and not run() because it retries failed tasks):
    # #------------------------------------------------------------------------------------------------------------
    # length = 1 #len(braket_circuits)
    # number_of_shots = metadata["number_of_repetitions"]
    # print(f"Will process {length} circuits")
    # print(f"Will process {length} circuits each one {number_of_shots} times")

    # circuits_batches = [braket_circuits[i:i + length] for i in range(0, len(braket_circuits), length)]


    # # [3] RUN
    # count = 0
    # #------------------------------------------------------------------------------------------------------------
    # # [3.1] Running batches with single circuits // tasks should be retried according to docs/Joanna
    # for batch in circuits_batches:
    #     print(f"Running  batch #{count}")
    #     count = count+1
    #      # [3.2] Run batch
    #     aws_batch = device.run_batch(task_specifications=batch,
    #                                  shots=number_of_shots,
    #                                  disable_qubit_rewiring=True)
        
    #     print(f"Created  #{len(aws_batch.tasks)} tasks for batch #{count}")


    #      # [3.3] Save task ID. In future - save results if not redundant?
    #     for task in aws_batch.tasks:
    #         print(f"task.id: {task.id}")
    #         append_new_line(task_arns_file, task.id)

    #         #Comm
    #         #results.append(task.result().measurement_counts)  #ADD
    #         #task_ids_list.append(task.id)
    #         #circuit_ids_list.append(count)

    #         pass    #endmark
    #     pass    #endmark
    
    #     results = aws_batch.result() # ALSO WILL RETRY FAILED TASKS


    # [4] Save job result - oprional, but may be heavy due to measurement_counts amount
    #save_job_result({ "measurement_counts": results, "task_id_list": task_ids_list, "circuit_id_list": circuit_ids_list})  #ADD
