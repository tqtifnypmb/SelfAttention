//
//  ContentView.swift
//  SelfAttention
//
//  Created by Tqtifnypmb on 2024/3/17.
//

import SwiftUI
import CoreML

final class Store: ObservableObject {
    
    let x: MLMultiArray
    let max_seq_len = 128
    let vocab_size = 10000
    let n_state = 384
    init() {
        self.x = try! MLMultiArray(shape: [1, self.max_seq_len as NSNumber], dataType: .int32)
        for idx in 0 ..< self.max_seq_len {
            self.x[[0, idx as NSNumber]] = Int.random(in: 0 ..< self.vocab_size) as NSNumber
        }
    }
    
    func run() {
        let (lhs, lhs_key_cache, _) = run_without_cache()
        let (rhs, rhs_key_cache, _) = run_with_cache()
        
        for idx in 0 ..< self.n_state {
            let lhs_y = lhs[[0, (lhs.shape[1].intValue - 1) as NSNumber, idx as NSNumber]]
            let rhs_y = rhs[[0, (rhs.shape[1].intValue - 1) as NSNumber, idx as NSNumber]]
            assert(abs(lhs_y.floatValue - rhs_y.floatValue) < 0.005)
        }
        
        for seq_idx in 0 ..< (self.max_seq_len - 1) {
            for state_idx in 0 ..< self.n_state {
                let lhs = lhs_key_cache[[0, seq_idx as NSNumber, state_idx as NSNumber]].floatValue
                
                // `(1 + seq_idx)` to ignore the padding
                let rhs = rhs_key_cache[[0, (1 + seq_idx) as NSNumber, state_idx as NSNumber]].floatValue
                assert(abs(lhs - rhs) < 0.005)
            }
        }
    }
    
    private func run_with_cache() -> (y: MLMultiArray, key_cache: MLMultiArray, value_cahce: MLMultiArray) {
        let model = try! self_attention_with_cache()
        
        let key_cache = try! MLMultiArray(shape: [1, self.max_seq_len as NSNumber, self.n_state as NSNumber], dataType: .float16)
        let value_cache = try! MLMultiArray(shape: [1, self.max_seq_len as NSNumber, self.n_state as NSNumber], dataType: .float16)
        for seq_idx in 0 ..< self.max_seq_len {
            let sliced_x = self.x[[0, seq_idx as NSNumber]]
            let input = try! MLMultiArray(shape: [1, 1], dataType: .int32)
            input[[0, 0]] = sliced_x
            let seq_len = try! MLMultiArray(shape: [1, 1], dataType: .int32)
            seq_len[[0, 0]] = (seq_idx + 1) as NSNumber
            
            let results = try! model.prediction(x: input, seq_len: seq_len, key_cache: key_cache, value_cache: value_cache)
            
            let y = results.featureValue(for: "y")!.multiArrayValue!
            let new_k_col = results.featureValue(for: "new_k_col")!.multiArrayValue!
            let new_v_col = results.featureValue(for: "new_v_col")!.multiArrayValue!
            if seq_idx == self.max_seq_len - 1 {
                return (y, key_cache, value_cache)
            } else {
                // update key_value cache
                for state_idx in 0 ..< self.n_state {
                    let index: [NSNumber] = [0, (seq_idx + 1) as NSNumber, state_idx as NSNumber]
                    
                    key_cache[index] = new_k_col[[0, 0, state_idx as NSNumber]]
                    value_cache[index] = new_v_col[[0, 0, state_idx as NSNumber]]
                }
            }
        }
        
        fatalError()
    }
    
    private func run_without_cache() -> (y: MLMultiArray, key_cache: MLMultiArray, value_cahce: MLMultiArray) {
        let model = try! self_attention_without_cache()
        let results = try! model.prediction(x: self.x)
        
        let y = results.featureValue(for: "y")!.multiArrayValue!
        let key_cache = results.featureValue(for: "key_cache")!.multiArrayValue!
        let value_cache = results.featureValue(for: "value_cache")!.multiArrayValue!
        return (y, key_cache, value_cache)
    }
}

struct ContentView: View {
    
    @StateObject
    private var store = Store()
    
    var body: some View {
        VStack {
            Image(systemName: "globe")
                .imageScale(.large)
                .foregroundStyle(.tint)
            Text("Hello, world!")
        }
        .padding()
        .onAppear {
            self.store.run()
        }
    }
}

#Preview {
    ContentView()
}
