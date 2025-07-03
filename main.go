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

	// Inicijalizuj ONNX runtime
	ort.SetSharedLibraryPath("/home/janko/onnxruntime-linux-x64-1.22.0/lib/libonnxruntime.so")
	if err := ort.InitializeEnvironment(); err != nil {
		log.Fatalf("Failed to init ONNX environment: %v", err)
	}
	defer ort.DestroyEnvironment()

	// Napravi ulazni tensor
	inputShape := ort.NewShape(1, int64(len(tokenIDs))) // npr. [1, dužina sekvence]
	inputTensor, err := ort.NewTensor(inputShape, tokenIDs)
	if err != nil {
		log.Fatalf("Failed to create input tensor: %v", err)
	}
	defer inputTensor.Destroy()

	// Pripremi izlazni tensor (dimenzije prema modelu)
	outputShape := ort.NewShape(1, int64(len(tokenIDs)), 384) // npr. [1, dužina, 384]
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		log.Fatalf("Failed to create output tensor: %v", err)
	}
	defer outputTensor.Destroy()

	// Pokreni sesiju
	inputNames := []string{"input_ids"}
	outputNames := []string{"last_hidden_state"}

	session, err := ort.NewAdvancedSession(
		"/home/janko/all-MiniLM-L6-v2/model.onnx",
		inputNames,
		outputNames,
		[]ort.Value{inputTensor},
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

	// Prikaži prvih nekoliko rezultata
	outputData := outputTensor.GetData()
	fmt.Printf("Output first 10 floats: %v\n", outputData[:10])
}
