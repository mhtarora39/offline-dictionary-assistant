package com.offline.dictionary.db

import android.content.Context
import android.database.sqlite.SQLiteDatabase
import java.io.File
import java.io.FileOutputStream
import kotlin.math.min

data class DefinitionItem(
    val headword: String,
    val pos: String,
    val definition: String,
    val example: String?
)

data class LookupResult(
    val resolution: String, // "exact", "fuzzy", "none"
    val resolvedHeadword: String?,
    val definitions: List<DefinitionItem>,
    val closestMatches: List<String> = emptyList(), // "Did you mean?" suggestions
    val relatedTerms: List<String> = emptyList()    // Related forms, compounds & derived entries
)

class DictionaryDbHelper(private val context: Context) {

    private val dbFile: File by lazy {
        val file = File(context.filesDir, "dictionary.db")
        val expectedSize = 45260800L
        if (!file.exists() || file.length() != expectedSize) {
            val tempFile = File(context.filesDir, "dictionary.db.tmp")
            context.assets.open("dictionary.db").use { input ->
                FileOutputStream(tempFile).use { output ->
                    input.copyTo(output)
                }
            }
            if (file.exists()) file.delete()
            tempFile.renameTo(file)
        }
        file
    }

    private val database: SQLiteDatabase by lazy {
        SQLiteDatabase.openDatabase(dbFile.path, null, SQLiteDatabase.OPEN_READONLY)
    }

    private fun normalize(term: String): String {
        val trimmed = term.trim().trim('\'', '"', '`', '“', '”', '‘', '’', '.', ',', '!', '?', ':', ';', '(', ')', '[', ']', '{', '}', '*', '~', '_', '#')
        return trimmed.lowercase().replace('_', ' ').replace(Regex("\\s+"), " ")
    }

    private val exactQuery = """
        SELECT e.headword, e.pos, s.gloss, s.example
        FROM forms f
        JOIN entries e ON e.id = f.entry_id
        JOIN senses s ON s.entry_id = e.id
        WHERE f.form = ?
        ORDER BY s.freq DESC, s.idx ASC
        LIMIT 10
    """.trimIndent()

    fun lookup(term: String): LookupResult {
        val normalized = normalize(term)
        if (normalized.isEmpty()) {
            return LookupResult("none", null, emptyList())
        }

        // 1. Exact lookup using forms index
        val exactResults = queryDefinitions(normalized)
        if (exactResults.isNotEmpty()) {
            val headword = exactResults.first().headword
            val related = getRelatedTerms(headword)
            return LookupResult(
                resolution = "exact",
                resolvedHeadword = headword,
                definitions = exactResults,
                closestMatches = emptyList(),
                relatedTerms = related
            )
        }

        // 2. Fuzzy fallback & closest match recommendations
        val closest = findClosestTerms(normalized, limit = 5)

        if (closest.isNotEmpty()) {
            val bestCandidate = closest.first()
            val bestDist = levenshtein(normalized, bestCandidate)

            if (bestDist <= 2) {
                val fuzzyResults = queryDefinitions(bestCandidate)
                if (fuzzyResults.isNotEmpty()) {
                    val headword = fuzzyResults.first().headword
                    val otherSuggestions = closest.filter { !it.equals(bestCandidate, ignoreCase = true) }
                    val related = getRelatedTerms(headword)
                    return LookupResult(
                        resolution = "fuzzy",
                        resolvedHeadword = headword,
                        definitions = fuzzyResults,
                        closestMatches = otherSuggestions,
                        relatedTerms = related
                    )
                }
            }

            // Word itself wasn't resolvable directly, but we have recommendations!
            return LookupResult(
                resolution = "none",
                resolvedHeadword = null,
                definitions = emptyList(),
                closestMatches = closest,
                relatedTerms = emptyList()
            )
        }

        return LookupResult("none", null, emptyList())
    }

    fun findClosestTerms(term: String, limit: Int = 4): List<String> {
        val normalized = normalize(term)
        if (normalized.length < 2) return emptyList()

        val termLen = normalized.length
        val minLen = (termLen - 3).coerceAtLeast(1)
        val maxLen = termLen + 3
        val candidates = LinkedHashSet<String>()

        // 1. Direct single-edit permutations (Transpositions, deletions, doubled chars, vowel mutations)
        // Evaluated via an indexed IN clause in ~0.5ms
        val edits = ArrayList<String>()
        val vowels = "aeiou"
        for (i in 0 until termLen - 1) {
            edits.add(normalized.substring(0, i) + normalized[i + 1] + normalized[i] + normalized.substring(i + 2))
        }
        for (i in 0 until termLen) {
            edits.add(normalized.substring(0, i) + normalized.substring(i + 1))
        }
        for (i in 0 until termLen) {
            edits.add(normalized.substring(0, i) + normalized[i] + normalized.substring(i))
        }
        for (i in 0 until termLen) {
            val ch = normalized[i]
            if (ch in vowels) {
                for (v in vowels) {
                    if (v != ch) {
                        edits.add(normalized.substring(0, i) + v + normalized.substring(i + 1))
                    }
                }
            }
        }

        if (edits.isNotEmpty()) {
            val placeholders = edits.joinToString(",") { "?" }
            val editsSql = "SELECT DISTINCT form FROM forms WHERE form IN ($placeholders)"
            database.rawQuery(editsSql, edits.toTypedArray()).use { cursor ->
                while (cursor.moveToNext()) {
                    candidates.add(cursor.getString(0))
                }
            }
        }

        // 2. Multi-level prefix scan with integer bounds (avoids SQLite string-comparison bug)
        if (normalized.contains(' ')) {
            val firstWord = normalized.substringBefore(' ')
            val prefixSql = """
                SELECT DISTINCT form FROM forms
                WHERE form LIKE ? AND length(form) BETWEEN $minLen AND $maxLen
                LIMIT 500
            """.trimIndent()
            database.rawQuery(prefixSql, arrayOf("$firstWord%")).use { cursor ->
                while (cursor.moveToNext()) {
                    candidates.add(cursor.getString(0))
                }
            }
        } else {
            val startLen = minOf(5, termLen)
            for (plen in startLen downTo 2) {
                val pref = normalized.take(plen)
                val prefNext = pref.dropLast(1) + (pref.last().code + 1).toChar()
                val rangeSql = """
                    SELECT DISTINCT form FROM forms
                    WHERE form >= ? AND form < ? AND length(form) BETWEEN $minLen AND $maxLen
                    LIMIT 400
                """.trimIndent()
                database.rawQuery(rangeSql, arrayOf(pref, prefNext)).use { cursor ->
                    while (cursor.moveToNext()) {
                        candidates.add(cursor.getString(0))
                    }
                }
                if (candidates.size >= 40) break
            }
        }

        data class ScoredCandidate(val term: String, val dist: Int)
        val scored = ArrayList<ScoredCandidate>()
        val maxAllowedDist = if (termLen <= 4) 2 else 3

        for (cand in candidates) {
            if (cand.equals(normalized, ignoreCase = true)) continue
            val d = levenshtein(normalized, cand)
            if (d <= maxAllowedDist) {
                scored.add(ScoredCandidate(cand, d))
            }
        }

        scored.sortWith(compareBy({ it.dist }, { kotlin.math.abs(it.term.length - termLen) }, { it.term }))

        val result = ArrayList<String>()
        val seen = HashSet<String>()
        for (item in scored) {
            if (seen.add(item.term)) {
                result.add(item.term)
                if (result.size >= limit) break
            }
        }
        return result
    }

    fun getRelatedTerms(headword: String, limit: Int = 8): List<String> {
        val normalized = normalize(headword)
        if (normalized.length < 2) return emptyList()

        val related = LinkedHashSet<String>()

        // 1. Phrases and compound idioms starting with headword (e.g. run across, run away, bucket seat)
        val phrasePrefix = "$normalized "
        val nextPhrasePrefix = "$normalized!"
        val phraseSql = """
            SELECT DISTINCT headword FROM entries
            WHERE headword >= ? AND headword < ?
            LIMIT 10
        """.trimIndent()
        database.rawQuery(phraseSql, arrayOf(phrasePrefix, nextPhrasePrefix)).use { cursor ->
            while (cursor.moveToNext()) {
                val phrase = cursor.getString(0)
                if (!phrase.equals(normalized, ignoreCase = true)) {
                    related.add(phrase)
                }
            }
        }

        // 2. Direct headwords starting with normalized term (e.g. kind -> kind of, kindhearted, kindly)
        val termPrefix = normalized
        val nextTermPrefix = termPrefix.dropLast(1) + (termPrefix.last().code + 1).toChar()
        val directSql = """
            SELECT DISTINCT headword FROM entries
            WHERE headword >= ? AND headword < ? AND headword != ?
            LIMIT 10
        """.trimIndent()
        database.rawQuery(directSql, arrayOf(termPrefix, nextTermPrefix, normalized)).use { cursor ->
            while (cursor.moveToNext()) {
                val hw = cursor.getString(0)
                if (!hw.equals(normalized, ignoreCase = true)) {
                    related.add(hw)
                }
            }
        }

        // 3. Morphological root variants (e.g. happy -> happiness, beauty -> beautiful, courage -> courageous)
        if (related.size < limit && normalized.length >= 4) {
            val root = when {
                normalized.endsWith("y") || normalized.endsWith("e") -> normalized.dropLast(1)
                normalized.length >= 6 -> normalized.dropLast(2)
                else -> normalized.dropLast(1)
            }
            if (root.isNotEmpty()) {
                val nextRoot = root.dropLast(1) + (root.last().code + 1).toChar()
                val rootSql = """
                    SELECT DISTINCT headword FROM entries
                    WHERE headword >= ? AND headword < ? AND headword != ?
                    LIMIT 10
                """.trimIndent()
                database.rawQuery(rootSql, arrayOf(root, nextRoot, normalized)).use { cursor ->
                    while (cursor.moveToNext()) {
                        val hw = cursor.getString(0)
                        if (!hw.equals(normalized, ignoreCase = true) && hw.startsWith(root.take(3))) {
                            related.add(hw)
                        }
                    }
                }
            }
        }

        return related.take(limit)
    }

    private fun queryDefinitions(form: String): List<DefinitionItem> {
        val results = ArrayList<DefinitionItem>()
        database.rawQuery(exactQuery, arrayOf(form)).use { cursor ->
            while (cursor.moveToNext()) {
                val hw = cursor.getString(0)
                val pos = cursor.getString(1)
                val def = cursor.getString(2)
                val ex = if (cursor.isNull(3)) null else cursor.getString(3)
                results.add(DefinitionItem(hw, pos, def, ex))
            }
        }
        return results
    }

    private fun levenshtein(s1: String, s2: String): Int {
        val dp = Array(s1.length + 1) { IntArray(s2.length + 1) }
        for (i in 0..s1.length) dp[i][0] = i
        for (j in 0..s2.length) dp[0][j] = j

        for (i in 1..s1.length) {
            for (j in 1..s2.length) {
                val cost = if (s1[i - 1] == s2[j - 1]) 0 else 1
                dp[i][j] = min(min(dp[i - 1][j] + 1, dp[i][j - 1] + 1), dp[i - 1][j - 1] + cost)
            }
        }
        return dp[s1.length][s2.length]
    }
}
