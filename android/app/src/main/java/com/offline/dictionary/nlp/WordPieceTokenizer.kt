package com.offline.dictionary.nlp

import android.content.Context
import java.io.BufferedReader
import java.io.InputStreamReader

data class TokenResult(
    val inputIds: LongArray,
    val attentionMask: LongArray,
    val wordIds: List<Int?>,
    val charOffsets: List<Pair<Int, Int>>
)

class WordPieceTokenizer(context: Context, vocabFileName: String = "vocab.txt") {

    private val vocab = HashMap<String, Long>()
    private val unkId: Long
    private val clsId: Long
    private val sepId: Long

    init {
        context.assets.open(vocabFileName).use { inputStream ->
            BufferedReader(InputStreamReader(inputStream, Charsets.UTF_8)).useLines { lines ->
                lines.forEachIndexed { index, token ->
                    val clean = token.trim()
                    if (clean.isNotEmpty()) {
                        vocab[clean] = index.toLong()
                    }
                }
            }
        }
        unkId = vocab["[UNK]"] ?: 100L
        clsId = vocab["[CLS]"] ?: 101L
        sepId = vocab["[SEP]"] ?: 102L
    }

    fun encode(text: String): TokenResult {
        val inputIds = ArrayList<Long>()
        val wordIds = ArrayList<Int?>()
        val offsets = ArrayList<Pair<Int, Int>>()

        // [CLS]
        inputIds.add(clsId)
        wordIds.add(null)
        offsets.add(Pair(0, 0))

        val regex = Regex("""\w+(?:'\w+)?|[^\s\w]""")
        var wordCounter = 0

        for (match in regex.findAll(text)) {
            val word = match.value.lowercase()
            val start = match.range.first
            val end = match.range.last + 1

            var subStart = 0
            val subTokens = ArrayList<String>()
            var isBad = false

            while (subStart < word.length) {
                var subEnd = word.length
                var curSub: String? = null
                while (subStart < subEnd) {
                    var sub = word.substring(subStart, subEnd)
                    if (subStart > 0) sub = "##$sub"
                    if (vocab.containsKey(sub)) {
                        curSub = sub
                        break
                    }
                    subEnd--
                }
                if (curSub == null) {
                    isBad = true
                    break
                }
                subTokens.add(curSub)
                subStart = subEnd
            }

            if (isBad || subTokens.isEmpty()) {
                inputIds.add(unkId)
                wordIds.add(wordCounter)
                offsets.add(Pair(start, end))
            } else {
                for (sub in subTokens) {
                    inputIds.add(vocab[sub] ?: unkId)
                    wordIds.add(wordCounter)
                    offsets.add(Pair(start, end))
                }
            }
            wordCounter++
        }

        // [SEP]
        inputIds.add(sepId)
        wordIds.add(null)
        offsets.add(Pair(0, 0))

        val mask = LongArray(inputIds.size) { 1L }
        return TokenResult(inputIds.toLongArray(), mask, wordIds, offsets)
    }
}
