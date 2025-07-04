package main

import (
	"fmt"
	"log"

	"github.com/daulet/tokenizers"
	ort "github.com/yalue/onnxruntime_go"
)

func main() {
	// Učitaj tokenizer
	tk, err := tokenizers.FromFile("/home/janko/all-MiniLM-L6-v2/tokenizer.json")
	if err != nil {
		log.Fatalf("Failed to load tokenizer: %v", err)
	}
	defer tk.Close()

	// Tokenizuj tekst
	inputText := "Hello, how are you?"
	idsUint32, tokens := tk.Encode(inputText, true)
	fmt.Printf("Tokens: %v\n", tokens)

	// Konvertuj []uint32 u []int64
	tokenIDs := make([]int64, len(idsUint32))
	for i, id := range idsUint32 {
		tokenIDs[i] = int64(id)
	}

	// Kreiraj token_type_ids slice iste dužine, sve nule
	tokenTypeIDs := make([]int64, len(tokenIDs))

	// Kreiraj attention_mask slice iste dužine
	// vrednost 1 za svaki ne-PAD token, 0 za PAD (pretpostavljamo da je PAD token id 0)
	attentionMask := make([]int64, len(tokenIDs))
	for i, id := range tokenIDs {
		if id == 0 {
			attentionMask[i] = 0
		} else {
			attentionMask[i] = 1
		}
	}

	// Inicijalizuj ONNX runtime
	ort.SetSharedLibraryPath("/home/janko/onnxruntime-linux-x64-1.22.0/lib/libonnxruntime.so")
	if err := ort.InitializeEnvironment(); err != nil {
		log.Fatalf("Failed to init ONNX environment: %v", err)
	}
	defer ort.DestroyEnvironment()

	inputShape := ort.NewShape(1, int64(len(tokenIDs))) // [1, seq_len]

	inputTensor, err := ort.NewTensor(inputShape, tokenIDs)
	if err != nil {
		log.Fatalf("Failed to create input_ids tensor: %v", err)
	}
	defer inputTensor.Destroy()

	tokenTypeTensor, err := ort.NewTensor(inputShape, tokenTypeIDs)
	if err != nil {
		log.Fatalf("Failed to create token_type_ids tensor: %v", err)
	}
	defer tokenTypeTensor.Destroy()

	attentionMaskTensor, err := ort.NewTensor(inputShape, attentionMask)
	if err != nil {
		log.Fatalf("Failed to create attention_mask tensor: %v", err)
	}
	defer attentionMaskTensor.Destroy()

	// Pripremi izlazni tensor (dimenzije prema modelu)
	outputShape := ort.NewShape(1, int64(len(tokenIDs)), 384)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		log.Fatalf("Failed to create output tensor: %v", err)
	}
	defer outputTensor.Destroy()

	inputNames := []string{"input_ids", "token_type_ids", "attention_mask"}
	outputNames := []string{"last_hidden_state"}

	session, err := ort.NewAdvancedSession(
		"/home/janko/all-MiniLM-L6-v2/model.onnx",
		inputNames,
		outputNames,
		[]ort.Value{inputTensor, tokenTypeTensor, attentionMaskTensor},
		[]ort.Value{outputTensor},
		nil,
	)
	if err != nil {
		log.Fatalf("Failed to create ONNX session: %v", err)
	}
	defer session.Destroy()

	if err := session.Run(); err != nil {
		log.Fatalf("Failed to run ONNX session: %v", err)
	}

	outputData := outputTensor.GetData()
	fmt.Printf("Output first 10 floats: %v\n", outputData[:10])
}
