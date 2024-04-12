import torch
import torch.nn as nn

import pennylane as qml

class QLSTM(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 n_qubits=8,
                 num_layers=1, #퀀텀 레이어
                 batch_first=True,
                 return_sequences=False,
                 return_state=False,
                 backend="default.qubit",
                 device='cpu'):

        super(QLSTM, self).__init__()
        self.device= device
        self.n_inputs = input_size
        self.hidden_size = hidden_size
        self.concat_size = self.n_inputs + self.hidden_size
        self.n_qubits = hidden_size
        self.n_qlayers = num_layers
        self.backend = backend  # "default.qubit", "qiskit.basicaer", "qiskit.ibm"

        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.return_state = return_state

        # self.dev = qml.device("default.qubit", wires=self.n_qubits)
        # self.dev = qml.device('qiskit.basicaer', wires=self.n_qubits)
        # self.dev = qml.device('qiskit.ibm', wires=self.n_qubits)
        # use 'qiskit.ibmq' instead to run on hardware

        self.wires_forget = [f"wire_forget_{i}" for i in range(self.n_qubits)]
        self.wires_input = [f"wire_input_{i}" for i in range(self.n_qubits)]
        self.wires_update = [f"wire_update_{i}" for i in range(self.n_qubits)]
        self.wires_output = [f"wire_output_{i}" for i in range(self.n_qubits)]

        self.dev_forget = qml.device(self.backend, wires=self.wires_forget)
        self.dev_input = qml.device(self.backend, wires=self.wires_input)
        self.dev_update = qml.device(self.backend, wires=self.wires_update)
        self.dev_output = qml.device(self.backend, wires=self.wires_output)

        self.qlayer_forget = qml.QNode(self._circuit_forget, self.dev_forget, interface="torch")
        self.qlayer_input = qml.QNode(self._circuit_input, self.dev_input, interface="torch")
        self.qlayer_update = qml.QNode(self._circuit_update, self.dev_update, interface="torch")
        self.qlayer_output = qml.QNode(self._circuit_output, self.dev_output, interface="torch")

        weight_shapes = {"weights": (num_layers, self.n_qubits)}
        print(f"weight_shapes = (n_qlayers, n_qubits) = ({num_layers}, {self.n_qubits})")

        self.clayer_in = torch.nn.Linear(self.concat_size, self.n_qubits).to(self.device)
        self.VQC = {
            'forget': qml.qnn.TorchLayer(self.qlayer_forget, weight_shapes),
            'input': qml.qnn.TorchLayer(self.qlayer_input, weight_shapes),
            'update': qml.qnn.TorchLayer(self.qlayer_update, weight_shapes),
            'output': qml.qnn.TorchLayer(self.qlayer_output, weight_shapes)
        }
        self.clayer_out = torch.nn.Linear(self.n_qubits, self.hidden_size).to(self.device)
        # self.clayer_out = [torch.nn.Linear(self.n_qubits, self.hidden_size) for _ in range(4)]

        self.to(device)



    def _circuit_output(self,inputs, weights):
        qml.templates.AngleEmbedding(inputs, wires=self.wires_output)
        qml.templates.BasicEntanglerLayers(weights, wires=self.wires_output)
        return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_output]

    def _circuit_update(self,inputs, weights):
        qml.templates.AngleEmbedding(inputs, wires=self.wires_update)
        qml.templates.BasicEntanglerLayers(weights, wires=self.wires_update)
        return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_update]

    def _circuit_forget(self,inputs, weights):
        qml.templates.AngleEmbedding(inputs, wires=self.wires_forget)
        qml.templates.BasicEntanglerLayers(weights, wires=self.wires_forget)
        return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_forget]

    def _circuit_input(self,inputs, weights):
        qml.templates.AngleEmbedding(inputs, wires=self.wires_input)
        qml.templates.BasicEntanglerLayers(weights, wires=self.wires_input)
        return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_input]

    def forward(self, x, init_states=None):
        '''
        x.shape is (batch_size, seq_length, feature_size)
        recurrent_activation -> sigmoid
        activation -> tanh
        '''
        if self.batch_first is True:
            batch_size, seq_length, features_size = x.size()
        else:
            seq_length, batch_size, features_size = x.size()

        hidden_seq = []
        if init_states is None:
            h_t = torch.zeros(batch_size, self.hidden_size).to(self.device)  # hidden state (output)
            c_t = torch.zeros(batch_size, self.hidden_size).to(self.device)  # cell state
        else:
            # for now we ignore the fact that in PyTorch you can stack multiple RNNs
            # so we take only the first elements of the init_states tuple init_states[0][0], init_states[1][0]
            h_t, c_t = init_states
            h_t = h_t[0].to(self.device)
            c_t = c_t[0].to(self.device)

        for t in range(seq_length):
            # get features from the t-th element in seq, for all entries in the batch
            x_t = x[:, t, :]

            # Concatenate input and hidden state

            v_t = torch.cat((h_t.to(self.device), x_t.to(self.device)), dim=1)

            # match qubit dimension
            y_t = self.clayer_in(v_t)

            f_t = torch.sigmoid(self.clayer_out(self.VQC['forget'](y_t))).to(self.device)  # forget block
            i_t = torch.sigmoid(self.clayer_out(self.VQC['input'](y_t))).to(self.device)  # input block
            g_t = torch.tanh(self.clayer_out(self.VQC['update'](y_t))).to(self.device)  # update block
            o_t = torch.sigmoid(self.clayer_out(self.VQC['output'](y_t))).to(self.device)  # output block

            c_t = (f_t * c_t) + (i_t * g_t)
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0).to(self.device))
        hidden_seq = torch.cat(hidden_seq, dim=0).to(self.device)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()


        return hidden_seq, (h_t, c_t)






class BiQLSTM(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 n_qubits=4,
                 num_layers=1, #퀀텀 레이어
                 batch_first=True,
                 return_sequences=False,
                 return_state=False,
                 backend="default.qubit",
                 device='cpu'):

        super(BiQLSTM, self).__init__()

        self.device = device
        self.n_inputs = input_size
        self.hidden_size = hidden_size
        self.concat_size = self.n_inputs + self.hidden_size
        self.n_qubits = hidden_size
        self.num_layers= num_layers
        self.backend = backend  # "default.qubit", "qiskit.basicaer", "qiskit.ibm"

        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.return_state = return_state


        self.forward_lstm = QLSTM(input_size=self.n_inputs, hidden_size=self.hidden_size, num_layers=self.num_layers,
                             batch_first=True,device=device)
        self.backward_lstm = QLSTM(input_size=self.n_inputs, hidden_size=self.hidden_size, num_layers=self.num_layers,
                             batch_first=True,device=device)

    def forward(self, x, init_states=None):
        forward_out, forward_state = self.forward_lstm(x, init_states)
        backward_out, backward_state = self.backward_lstm(x.flip(1), init_states)  # flip input for backward pass
        out = torch.cat((forward_out, backward_out), dim=-1)
        state = (torch.cat((forward_state[0], backward_state[0]), dim=-1),
                 torch.cat((forward_state[1], backward_state[1]), dim=-1))
        return out, state

