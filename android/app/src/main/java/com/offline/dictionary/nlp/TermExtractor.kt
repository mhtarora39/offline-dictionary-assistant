package com.offline.dictionary.nlp

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.nio.LongBuffer

class TermExtractor(context: Context) {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val tokenizer: WordPieceTokenizer = WordPieceTokenizer(context)
    private val labelList = listOf("O", "B-TERM", "I-TERM")
    private val punctAndQuotes = "'\"`“”‘’.,!?:;()[]{}*~_#"

    init {
        val modelBytes = context.assets.open("term_extractor_int8.onnx").use { it.readBytes() }
        val sessionOptions = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(2)
        }
        session = env.createSession(modelBytes, sessionOptions)
    }

    data class WordSpan(val wordId: Int, val label: String, var start: Int, var end: Int)

    fun extract(query: String): String? {
        val trimmed = query.trim()
        if (trimmed.isEmpty()) return null

        // Direct single-word pass-through
        val singleWord = trimmed.trim { it in punctAndQuotes }
        if (!singleWord.contains(" ") && singleWord.isNotEmpty()) {
            return singleWord
        }

        val encoded = tokenizer.encode(trimmed)
        val seqLen = encoded.inputIds.size.toLong()
        val shape = longArrayOf(1, seqLen)

        val inputIdsTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(encoded.inputIds), shape)
        val maskTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(encoded.attentionMask), shape)

        val inputs = mapOf(
            "input_ids" to inputIdsTensor,
            "attention_mask" to maskTensor
        )

        val results = session.run(inputs)

        @Suppress("UNCHECKED_CAST")
        val rawLogits = (results[0].value as Array<Array<FloatArray>>)[0]

        // Argmax predictions
        val predIds = IntArray(encoded.inputIds.size) { i ->
            val l = rawLogits[i]
            var maxIdx = 0
            if (l[1] > l[0]) maxIdx = 1
            if (l[2] > l[maxIdx]) maxIdx = 2
            maxIdx
        }

        // Group wordpieces by wordId (take label strictly from FIRST subtoken)
        val words = ArrayList<WordSpan>()
        val seenWords = HashSet<Int>()

        for (i in encoded.wordIds.indices) {
            val wId = encoded.wordIds[i] ?: continue
            val (cStart, cEnd) = encoded.charOffsets[i]

            if (!seenWords.contains(wId)) {
                seenWords.add(wId)
                words.add(WordSpan(wId, labelList[predIds[i]], cStart, cEnd))
            } else {
                val last = words.last()
                last.end = maxOf(last.end, cEnd)
                last.start = minOf(last.start, cStart)
            }
        }

        if (words.isEmpty()) return null

        // Find first contiguous run of B-TERM / I-TERM
        var startIdx = -1
        for (i in words.indices) {
            if (words[i].label == "B-TERM" || words[i].label == "I-TERM") {
                startIdx = i
                break
            }
        }
        if (startIdx == -1) return null

        var endIdx = startIdx
        while (endIdx + 1 < words.size && (words[endIdx + 1].label == "B-TERM" || words[endIdx + 1].label == "I-TERM")) {
            endIdx++
        }

        val sliceStart = words[startIdx].start
        val sliceEnd = words[endIdx].end

        // Sliced directly from raw string -> 0.0% invented terms
        val rawSlice = trimmed.substring(sliceStart, sliceEnd)
        val cleaned = rawSlice.trim().trim { it in punctAndQuotes }.trim()

        return cleaned.ifEmpty { null }
    }

    fun close() {
        session.close()
        env.close()
    }
}
