"""
Esporta i pesi e i bias di una rete fully-connected allenata con PyTorch nel file binario
letto da LoadNetworkHeader / LoadNetworkWeights (network.c).

Formato del file (little endian, int32 e float32):
    int32   MAGIC_NUMBER
    int32   num_layers
    num_layers x { int32 input_size, int32 output_size }       <- header
    num_layers x { float32 weights[input_size * output_size],   <- row-major, shape (input, output)
                   float32 biases[output_size] }

Note importanti:
  * PyTorch salva i pesi di nn.Linear con shape (out, in); il C li vuole (in, out), quindi vengono trasposti.
  * Il C applica la ReLU dopo ogni layer tranne l'ultimo: la rete deve avere questa struttura
    (Linear -> ReLU -> Linear -> ... -> Linear), altrimenti l'export viene rifiutato.

Uso:
    python export_C_data.py model.pt                  # state_dict (caricamento sicuro, solo tensori)
    python export_C_data.py model.pt --trust          # modello intero (torch.save(model)): pickle, solo file fidati
    python export_C_data.py model.pt -o out.bin
    python export_C_data.py --demo                    # rete di esempio hardcoded (nessun PyTorch necessario)
"""
import argparse
import os
import struct
import sys

import numpy as np

MAGIC_NUMBER = 0x1404A715
DEFAULT_OUTPUT = "weights_biases.bin"

# Limiti di sicurezza: servono a rifiutare file palesemente anomali, non a limitare reti reali.
# Devono restare allineati con quelli di LoadNetworkHeader (network.c).
MAX_LAYERS = 256
MAX_DIM = 1 << 20                 # neuroni per layer
MAX_PARAMS = 1 << 28              # numero totale di float (~1 GB)
MAX_FILE_BYTES = 2 * 1024 ** 3    # dimensione massima del file di input


# --------------------------------------------------------------------------------------------
# Estrazione dei layer: ogni layer e' una tupla (weights[in, out], bias[out]) in float32
# --------------------------------------------------------------------------------------------
def layers_from_module(model):
    """Estrae i layer da un nn.Module, verificando che sia una sequenza Linear/ReLU compatibile col C."""
    import torch.nn as nn

    layers = []
    expect_activation = False          # dopo ogni Linear (tranne l'ultimo) deve esserci una ReLU
    ignored = (nn.Identity, nn.Dropout, nn.Flatten)   # no-op in inferenza

    for name, m in model.named_modules():
        if m is model and not isinstance(m, nn.Linear):
            continue
        if isinstance(m, nn.Linear):
            if expect_activation:
                raise ValueError(f"Manca la ReLU prima del layer '{name}': il C la applica sempre tra due layer.")
            w = m.weight.detach().cpu().numpy().T                      # (out, in) -> (in, out)
            b = (m.bias.detach().cpu().numpy() if m.bias is not None
                 else np.zeros(m.out_features, dtype=np.float32))
            layers.append((w, b))
            expect_activation = True
        elif isinstance(m, nn.ReLU):
            if not expect_activation:
                raise ValueError(f"ReLU inattesa in '{name}' (due ReLU consecutive o ReLU iniziale).")
            expect_activation = False
        elif isinstance(m, ignored) or len(list(m.children())) > 0:
            continue                    # no-op oppure container (Sequential, ...)
        else:
            raise ValueError(f"Modulo non supportato dal motore C: '{name}' ({type(m).__name__}).")

    if expect_activation is False and layers:
        raise ValueError("L'ultimo layer non deve essere seguito da una ReLU (il C non la applica).")
    return layers


def layers_from_state_dict(state):
    """Estrae i layer da uno state_dict: coppie '<prefisso>.weight' (2D) / '<prefisso>.bias' nell'ordine del dict.
    Le attivazioni non sono visibili: si assume lo schema Linear -> ReLU -> ... -> Linear."""
    layers = []
    for key, tensor in state.items():
        if not key.endswith("weight"):
            continue
        w = tensor.detach().cpu().numpy()
        if w.ndim != 2:
            raise ValueError(f"'{key}' non e' una matrice 2D (shape {w.shape}): solo layer Linear supportati.")
        bias_key = key[: -len("weight")] + "bias"
        if bias_key in state:
            b = state[bias_key].detach().cpu().numpy()
        else:
            b = np.zeros(w.shape[0], dtype=np.float32)
        layers.append((w.T, b))
    return layers


def load_layers(path, trust=False):
    """Di default carica solo tensori (weights_only=True): un .pt contenente codice pickle viene rifiutato.
    Con trust=True si possono caricare anche modelli interi, ma il pickle puo' eseguire codice arbitrario."""
    import torch
    import torch.nn as nn

    if not os.path.isfile(path):
        raise ValueError(f"File '{path}' non trovato.")
    if os.path.getsize(path) > MAX_FILE_BYTES:
        raise ValueError(f"File '{path}' troppo grande (> {MAX_FILE_BYTES} byte).")

    try:
        obj = torch.load(path, map_location="cpu", weights_only=not trust)
    except Exception as e:
        hint = "" if trust else " Se e' un modello intero di cui ti fidi, usa --trust (oppure salva lo state_dict)."
        raise ValueError(f"Impossibile caricare '{path}' in modo sicuro: {e}.{hint}")

    if isinstance(obj, nn.Module):
        obj.eval()
        return layers_from_module(obj)
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):   # checkpoint tipico
            obj = obj["state_dict"]
        return layers_from_state_dict(obj)
    raise ValueError(f"Contenuto di '{path}' non riconosciuto ({type(obj).__name__}).")


def demo_layers():
    """Rete di esempio 2 -> 3 -> 2 (pesi in forma (input, output))."""
    w1 = np.array([1, -2, 3, 4, 5, -6], dtype=np.float32).reshape(2, 3)
    b1 = np.array([-3, 2, -0.5], dtype=np.float32)
    w2 = np.array([6, 5.5, 4, -3, 2, 1], dtype=np.float32).reshape(3, 2)
    b2 = np.array([2, -2], dtype=np.float32)
    return [(w1, b1), (w2, b2)]


# --------------------------------------------------------------------------------------------
# Validazione, scrittura e verifica
# --------------------------------------------------------------------------------------------
def validate(layers):
    if not layers:
        raise ValueError("Nessun layer da esportare.")
    if len(layers) > MAX_LAYERS:
        raise ValueError(f"Troppi layer ({len(layers)} > {MAX_LAYERS}).")
    total = 0
    for i, (w, b) in enumerate(layers):
        if w.ndim != 2 or b.ndim != 1 or w.shape[1] != b.shape[0]:
            raise ValueError(f"Layer {i}: shape incoerenti (weights {w.shape}, bias {b.shape}).")
        if min(w.shape) < 1 or max(w.shape) > MAX_DIM:
            raise ValueError(f"Layer {i}: dimensioni fuori limite (weights {w.shape}, max {MAX_DIM}).")
        total += w.size + b.size
        if total > MAX_PARAMS:
            raise ValueError(f"Troppi parametri (> {MAX_PARAMS}).")
        if not (np.isfinite(w).all() and np.isfinite(b).all()):
            raise ValueError(f"Layer {i}: contiene NaN o inf.")
        if i > 0 and layers[i - 1][0].shape[1] != w.shape[0]:
            raise ValueError(f"Layer {i}: input_size {w.shape[0]} != output_size del layer precedente "
                             f"{layers[i - 1][0].shape[1]}.")


def write_binary(layers, path):
    """Scrive su un file temporaneo e lo rinomina a fine scrittura: un errore non lascia file a meta'."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<ii", MAGIC_NUMBER, len(layers)))
        for w, _ in layers:                                  # header: tutte le dimensioni prima dei dati
            f.write(struct.pack("<ii", w.shape[0], w.shape[1]))
        for w, b in layers:
            f.write(np.ascontiguousarray(w, dtype="<f4").tobytes())   # row-major (in, out)
            f.write(np.ascontiguousarray(b, dtype="<f4").tobytes())
    os.replace(tmp, path)


def read_binary(path):
    """Rilegge il file con la stessa logica del C: serve a verificare l'export."""
    with open(path, "rb") as f:
        magic, n = struct.unpack("<ii", f.read(8))
        if magic != MAGIC_NUMBER:
            raise ValueError("Magic number errato.")
        dims = [struct.unpack("<ii", f.read(8)) for _ in range(n)]
        layers = []
        for i_size, o_size in dims:
            w = np.frombuffer(f.read(4 * i_size * o_size), dtype="<f4").reshape(i_size, o_size)
            b = np.frombuffer(f.read(4 * o_size), dtype="<f4")
            layers.append((w, b))
        if f.read(1):
            raise ValueError("Byte in eccesso a fine file.")
    return layers


def forward(layers, x):
    """Riferimento numpy identico a ForwardPass del C (ReLU tra i layer, non sull'ultimo)."""
    for i, (w, b) in enumerate(layers):
        x = x @ w + b
        if i < len(layers) - 1:
            x = np.maximum(x, 0)
    return x


def verify(layers, path):
    back = read_binary(path)
    if len(back) != len(layers):
        raise ValueError("Numero di layer diverso dopo la rilettura.")
    for i, ((w, b), (w2, b2)) in enumerate(zip(layers, back)):
        if not (np.array_equal(w.astype(np.float32), w2) and np.array_equal(b.astype(np.float32), b2)):
            raise ValueError(f"Layer {i}: i dati riletti non coincidono con quelli esportati.")


def main():
    ap = argparse.ArgumentParser(description="Esporta una rete PyTorch nel formato binario del motore di inferenza C.")
    ap.add_argument("model", nargs="?", help="file .pt/.pth (state_dict; modello intero solo con --trust)")
    ap.add_argument("-o", "--output", default=DEFAULT_OUTPUT, help=f"file di output (default: {DEFAULT_OUTPUT})")
    ap.add_argument("--demo", action="store_true", help="esporta la rete di esempio hardcoded")
    ap.add_argument("--trust", action="store_true",
                    help="permette di caricare modelli interi (pickle, puo' eseguire codice): solo per file fidati")
    args = ap.parse_args()

    if args.demo == (args.model is not None):
        ap.error("specificare esattamente uno tra MODEL e --demo")

    try:
        if args.model and os.path.abspath(args.model) == os.path.abspath(args.output):
            raise ValueError("Il file di output coincide con il modello di input.")
        layers = demo_layers() if args.demo else load_layers(args.model, args.trust)
        layers = [(np.asarray(w, dtype=np.float32), np.asarray(b, dtype=np.float32)) for w, b in layers]
        validate(layers)
        write_binary(layers, args.output)
        verify(layers, args.output)
    except (ValueError, OSError, ImportError) as e:
        print(f"Errore: {e}", file=sys.stderr)
        sys.exit(1)

    sizes = " -> ".join([str(layers[0][0].shape[0])] + [str(w.shape[1]) for w, _ in layers])
    print(f"Esportati {len(layers)} layer ({sizes}) in '{args.output}'.")


if __name__ == "__main__":
    main()
