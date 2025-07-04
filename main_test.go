package main

import (
	"log"
	"testing"

	"github.com/daulet/tokenizers"
	ort "github.com/yalue/onnxruntime_go"
)

func BenchmarkONNXInference(b *testing.B) {
	tokenizer, err := tokenizers.FromFile("/home/janko/all-MiniLM-L6-v2/tokenizer.json")
	if err != nil {
		log.Fatalf("Failed to load tokenizer: %v", err)
	}
	defer tokenizer.Close()

	// Tokenizuj tekst
	text := "Hello, how are you?"
	idsUint32, _ := tokenizer.Encode(text, true)

	tokenIDs := make([]int64, len(idsUint32))
	for i, id := range idsUint32 {
		tokenIDs[i] = int64(id)
	}

	tokenTypeIDs := make([]int64, len(tokenIDs))
	attentionMask := make([]int64, len(tokenIDs))
	for i, id := range tokenIDs {
		if id == 0 {
			attentionMask[i] = 0
		} else {
			attentionMask[i] = 1
		}
	}

	// Inicijalizuj ONNX Runtime
	ort.SetSharedLibraryPath("/home/janko/onnxruntime-linux-x64-1.22.0/lib/libonnxruntime.so")
	if err := ort.InitializeEnvironment(); err != nil {
		log.Fatalf("Failed to initialize ONNX environment: %v", err)
	}
	defer ort.DestroyEnvironment()

	// Priprema tenzora
	inputShape := ort.NewShape(1, int64(len(tokenIDs)))

	inputTensor, err := ort.NewTensor(inputShape, tokenIDs)
	if err != nil {
		log.Fatalf("Failed to create input tensor: %v", err)
	}
	defer inputTensor.Destroy()

	tokenTypeTensor, err := ort.NewTensor(inputShape, tokenTypeIDs)
	if err != nil {
		log.Fatalf("Failed to create token type tensor: %v", err)
	}
	defer tokenTypeTensor.Destroy()

	attentionMaskTensor, err := ort.NewTensor(inputShape, attentionMask)
	if err != nil {
		log.Fatalf("Failed to create attention mask tensor: %v", err)
	}
	defer attentionMaskTensor.Destroy()

	outputShape := ort.NewShape(1, int64(len(tokenIDs)), 384)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		log.Fatalf("Failed to create output tensor: %v", err)
	}
	defer outputTensor.Destroy()

	// Kreiranje sesije
	session, err := ort.NewAdvancedSession(
		"/home/janko/all-MiniLM-L6-v2/model.onnx",
		[]string{"input_ids", "token_type_ids", "attention_mask"},
		[]string{"last_hidden_state"},
		[]ort.Value{inputTensor, tokenTypeTensor, attentionMaskTensor},
		[]ort.Value{outputTensor},
		nil,
	)
	if err != nil {
		log.Fatalf("Failed to create ONNX session: %v", err)
	}
	defer session.Destroy()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := session.Run(); err != nil {
			b.Fatalf("Failed to run ONNX session: %v", err)
		}
	}
}
