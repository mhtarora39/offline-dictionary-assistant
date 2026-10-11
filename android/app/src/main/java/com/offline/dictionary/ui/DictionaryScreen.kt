package com.offline.dictionary.ui

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Clear
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.MicOff
import androidx.compose.material.icons.filled.Search
import androidx.compose.material.icons.filled.VolumeOff
import androidx.compose.material.icons.filled.VolumeUp
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.offline.dictionary.db.DefinitionItem

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DictionaryScreen(
    viewModel: DictionaryViewModel,
    onMicClick: () -> Unit
) {
    val state by viewModel.uiState.collectAsState()

    Scaffold(
        containerColor = Color(0xFF0F172A),
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text(
                            text = "Offline Dictionary",
                            fontWeight = FontWeight.Bold,
                            fontSize = 20.sp,
                            color = Color.White
                        )
                        Text(
                            text = "100% On-Device AI • WordNet 3.0",
                            fontSize = 12.sp,
                            color = Color(0xFF94A3B8)
                        )
                    }
                },
                actions = {
                    FilledTonalButton(
                        onClick = { viewModel.toggleAutoSpeakVoiceResults() },
                        colors = ButtonDefaults.filledTonalButtonColors(
                            containerColor = if (state.autoSpeakVoiceResults) Color(0xFF0284C7).copy(alpha = 0.25f) else Color(0xFF334155),
                            contentColor = if (state.autoSpeakVoiceResults) Color(0xFF38BDF8) else Color(0xFF94A3B8)
                        ),
                        shape = RoundedCornerShape(20.dp),
                        contentPadding = PaddingValues(horizontal = 10.dp, vertical = 4.dp),
                        modifier = Modifier.padding(end = 8.dp)
                    ) {
                        Icon(
                            imageVector = if (state.autoSpeakVoiceResults) Icons.Default.VolumeUp else Icons.Default.VolumeOff,
                            contentDescription = "Auto-speak voice results",
                            modifier = Modifier.size(16.dp)
                        )
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(
                            text = if (state.autoSpeakVoiceResults) "Voice Read: ON" else "Voice Read: OFF",
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Medium
                        )
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = Color(0xFF1E293B)
                )
            )
        }
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(16.dp)
        ) {
            // Search Input Row
            OutlinedTextField(
                value = state.query,
                onValueChange = { viewModel.onQueryChanged(it) },
                modifier = Modifier.fillMaxWidth(),
                placeholder = {
                    Text(
                        text = if (state.isListening) "Listening... speak now" else "Ask anything (e.g. what is serendipity)",
                        color = if (state.isListening) Color(0xFF38BDF8) else Color(0xFF64748B),
                        fontWeight = if (state.isListening) FontWeight.Medium else FontWeight.Normal
                    )
                },
                leadingIcon = {
                    IconButton(onClick = { viewModel.performSearch() }) {
                        Icon(
                            imageVector = Icons.Default.Search,
                            contentDescription = "Search",
                            tint = Color(0xFF38BDF8)
                        )
                    }
                },
                trailingIcon = {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        if (state.query.isNotEmpty()) {
                            IconButton(onClick = { viewModel.onQueryChanged("") }) {
                                Icon(
                                    imageVector = Icons.Default.Clear,
                                    contentDescription = "Clear",
                                    tint = Color(0xFF94A3B8)
                                )
                            }
                        }
                        IconButton(
                            onClick = { onMicClick() },
                            modifier = Modifier
                                .clip(CircleShape)
                                .background(if (state.isListening) Color(0xFFEF4444) else Color(0xFF334155))
                        ) {
                            Icon(
                                imageVector = if (state.isListening) Icons.Default.MicOff else Icons.Default.Mic,
                                contentDescription = "Voice Search",
                                tint = Color.White
                            )
                        }
                    }
                },
                singleLine = true,
                shape = RoundedCornerShape(16.dp),
                colors = OutlinedTextFieldDefaults.colors(
                    focusedContainerColor = Color(0xFF1E293B),
                    unfocusedContainerColor = Color(0xFF1E293B),
                    focusedTextColor = Color.White,
                    unfocusedTextColor = Color.White,
                    focusedBorderColor = Color(0xFF38BDF8),
                    unfocusedBorderColor = Color(0xFF334155)
                ),
                keyboardOptions = KeyboardOptions(imeAction = ImeAction.Search),
                keyboardActions = KeyboardActions(
                    onSearch = { viewModel.performSearch() },
                    onDone = { viewModel.performSearch() },
                    onGo = { viewModel.performSearch() }
                )
            )

            Spacer(modifier = Modifier.height(12.dp))

            // Latency / Status pill
            AnimatedVisibility(visible = state.lookupResult != null) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    if (state.extractedTerm != null) {
                        Text(
                            text = "Extracted: \"${state.extractedTerm}\"",
                            color = Color(0xFF38BDF8),
                            fontSize = 13.sp,
                            fontWeight = FontWeight.Medium
                        )
                    }
                    Text(
                        text = "⚡ ${state.extractionTimeMs}ms NLP • ${state.totalTimeMs}ms total",
                        color = Color(0xFF64748B),
                        fontSize = 12.sp,
                        fontFamily = FontFamily.Monospace
                    )
                }
            }

            // Error notice
            if (state.errorMessage != null) {
                Spacer(modifier = Modifier.height(8.dp))
                Surface(
                    color = Color(0xFF451A1A),
                    shape = RoundedCornerShape(8.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(
                        text = state.errorMessage ?: "",
                        color = Color(0xFFFCA5A5),
                        modifier = Modifier.padding(12.dp),
                        fontSize = 13.sp
                    )
                }
            }

            Spacer(modifier = Modifier.height(8.dp))

            // Results List
            val res = state.lookupResult
            if (res != null) {
                if (res.resolution == "none" || res.definitions.isEmpty()) {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(top = 24.dp),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        Text(
                            text = "No definition found",
                            color = Color.White,
                            fontSize = 18.sp,
                            fontWeight = FontWeight.SemiBold
                        )
                        Spacer(modifier = Modifier.height(6.dp))
                        Text(
                            text = if (state.extractedTerm != null) "No exact match for '${state.extractedTerm}'" else "Try rephrasing your question",
                            color = Color(0xFF94A3B8),
                            fontSize = 14.sp
                        )

                        // Recommendations / Did you mean?
                        if (res.closestMatches.isNotEmpty()) {
                            Spacer(modifier = Modifier.height(24.dp))
                            Surface(
                                color = Color(0xFF1E293B),
                                shape = RoundedCornerShape(16.dp),
                                border = androidx.compose.foundation.BorderStroke(1.dp, Color(0xFF38BDF8).copy(alpha = 0.35f)),
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                Column(modifier = Modifier.padding(16.dp)) {
                                    Row(verticalAlignment = Alignment.CenterVertically) {
                                        Icon(
                                            imageVector = Icons.Default.Search,
                                            contentDescription = null,
                                            tint = Color(0xFF38BDF8),
                                            modifier = Modifier.size(18.dp)
                                        )
                                        Spacer(modifier = Modifier.width(8.dp))
                                        Text(
                                            text = "Did you mean?",
                                            fontWeight = FontWeight.Bold,
                                            fontSize = 15.sp,
                                            color = Color.White
                                        )
                                    }
                                    Spacer(modifier = Modifier.height(6.dp))
                                    Text(
                                        text = "Tap a suggested word to view its definition:",
                                        fontSize = 12.sp,
                                        color = Color(0xFF94A3B8)
                                    )
                                    Spacer(modifier = Modifier.height(12.dp))
                                    SuggestionChipsRow(
                                        suggestions = res.closestMatches,
                                        chipColor = Color(0xFF0369A1),
                                        textColor = Color.White,
                                        onChipClick = { viewModel.performSearch(it) }
                                    )
                                }
                            }
                        }
                    }
                } else {
                    LazyColumn(
                        modifier = Modifier.fillMaxSize(),
                        verticalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        // Fuzzy banner & other suggestions
                        if (res.resolution == "fuzzy") {
                            item {
                                Surface(
                                    color = Color(0xFF1E293B),
                                    shape = RoundedCornerShape(12.dp),
                                    border = androidx.compose.foundation.BorderStroke(1.dp, Color(0xFFFBBF24).copy(alpha = 0.4f)),
                                    modifier = Modifier.fillMaxWidth()
                                ) {
                                    Column(modifier = Modifier.padding(12.dp)) {
                                        Text(
                                            text = "Showing results for \"${res.resolvedHeadword}\" (you typed \"${state.query}\")",
                                            color = Color(0xFFFBBF24),
                                            fontSize = 13.sp,
                                            fontWeight = FontWeight.Medium
                                        )
                                        if (res.closestMatches.isNotEmpty()) {
                                            Spacer(modifier = Modifier.height(8.dp))
                                            Text(
                                                text = "Other suggestions:",
                                                color = Color(0xFF94A3B8),
                                                fontSize = 11.sp
                                            )
                                            Spacer(modifier = Modifier.height(6.dp))
                                            SuggestionChipsRow(
                                                suggestions = res.closestMatches,
                                                chipColor = Color(0xFF334155),
                                                textColor = Color(0xFFFBBF24),
                                                onChipClick = { viewModel.performSearch(it) }
                                            )
                                        }
                                    }
                                }
                            }
                        }

                        // Definition Cards
                        itemsIndexed(res.definitions) { index, item ->
                            DefinitionCard(
                                item = item,
                                senseIndex = index + 1,
                                isSpeaking = state.isSpeaking,
                                currentSpeakingId = state.currentSpeakingId,
                                onSpeakClick = { utteranceId ->
                                    viewModel.speakDefinition(
                                        headword = item.headword,
                                        pos = item.pos,
                                        definition = item.definition,
                                        example = item.example,
                                        utteranceId = utteranceId
                                    )
                                }
                            )
                        }

                        // Related Terms & Expressions Section
                        if (res.relatedTerms.isNotEmpty()) {
                            item {
                                Surface(
                                    color = Color(0xFF1E293B),
                                    shape = RoundedCornerShape(14.dp),
                                    border = androidx.compose.foundation.BorderStroke(1.dp, Color(0xFF334155)),
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .padding(vertical = 4.dp)
                                ) {
                                    Column(modifier = Modifier.padding(14.dp)) {
                                        Text(
                                            text = "🔗 Related Words & Phrases",
                                            fontWeight = FontWeight.Bold,
                                            fontSize = 14.sp,
                                            color = Color.White
                                        )
                                        Spacer(modifier = Modifier.height(4.dp))
                                        Text(
                                            text = "Tap any term to explore definitions:",
                                            fontSize = 11.sp,
                                            color = Color(0xFF94A3B8)
                                        )
                                        Spacer(modifier = Modifier.height(10.dp))
                                        SuggestionChipsRow(
                                            suggestions = res.relatedTerms,
                                            chipColor = Color(0xFF334155),
                                            textColor = Color(0xFF38BDF8),
                                            onChipClick = { viewModel.performSearch(it) }
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun SuggestionChipsRow(
    suggestions: List<String>,
    chipColor: Color,
    textColor: Color,
    onChipClick: (String) -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .horizontalScroll(rememberScrollState()),
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        suggestions.forEach { suggestion ->
            Surface(
                onClick = { onChipClick(suggestion) },
                color = chipColor,
                shape = RoundedCornerShape(20.dp),
                border = androidx.compose.foundation.BorderStroke(1.dp, textColor.copy(alpha = 0.35f))
            ) {
                Row(
                    modifier = Modifier.padding(horizontal = 14.dp, vertical = 7.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = suggestion,
                        color = textColor,
                        fontSize = 13.sp,
                        fontWeight = FontWeight.Medium
                    )
                }
            }
        }
    }
}

@Composable
fun DefinitionCard(
    item: DefinitionItem,
    senseIndex: Int,
    isSpeaking: Boolean,
    currentSpeakingId: String?,
    onSpeakClick: (String) -> Unit
) {
    val utteranceId = "card_${item.headword}_${item.pos}_${senseIndex}"
    val isThisCardSpeaking = isSpeaking && currentSpeakingId == utteranceId

    val posBadgeColor = when (item.pos.lowercase()) {
        "noun" -> Color(0xFF0284C7)
        "verb" -> Color(0xFF16A34A)
        "adj" -> Color(0xFF9333EA)
        "adv" -> Color(0xFFD97706)
        else -> Color(0xFF475569)
    }

    Surface(
        color = Color(0xFF1E293B),
        shape = RoundedCornerShape(12.dp),
        border = androidx.compose.foundation.BorderStroke(
            if (isThisCardSpeaking) 1.5.dp else 1.dp,
            if (isThisCardSpeaking) Color(0xFF38BDF8) else Color(0xFF334155).copy(alpha = 0.6f)
        ),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    text = item.headword,
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color.White
                )
                Spacer(modifier = Modifier.width(8.dp))
                Surface(
                    color = posBadgeColor.copy(alpha = 0.2f),
                    border = androidx.compose.foundation.BorderStroke(1.dp, posBadgeColor.copy(alpha = 0.6f)),
                    shape = RoundedCornerShape(6.dp)
                ) {
                    Text(
                        text = item.pos.uppercase(),
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Bold,
                        color = posBadgeColor,
                        modifier = Modifier.padding(horizontal = 7.dp, vertical = 2.dp)
                    )
                }
                Spacer(modifier = Modifier.weight(1f))

                // Speaker Icon Button
                IconButton(
                    onClick = { onSpeakClick(utteranceId) },
                    modifier = Modifier
                        .size(32.dp)
                        .clip(CircleShape)
                        .background(
                            if (isThisCardSpeaking) Color(0xFF0284C7).copy(alpha = 0.35f)
                            else Color(0xFF0F172A).copy(alpha = 0.5f)
                        )
                ) {
                    Icon(
                        imageVector = Icons.Default.VolumeUp,
                        contentDescription = "Speak definition",
                        tint = if (isThisCardSpeaking) Color(0xFF38BDF8) else Color(0xFF94A3B8),
                        modifier = Modifier.size(17.dp)
                    )
                }

                Spacer(modifier = Modifier.width(6.dp))

                Text(
                    text = "#$senseIndex",
                    fontSize = 11.sp,
                    color = Color(0xFF64748B),
                    fontWeight = FontWeight.SemiBold
                )
            }

            Spacer(modifier = Modifier.height(8.dp))

            Text(
                text = item.definition,
                fontSize = 15.sp,
                color = Color(0xFFE2E8F0),
                lineHeight = 21.sp
            )

            if (!item.example.isNullOrBlank()) {
                Spacer(modifier = Modifier.height(8.dp))
                Surface(
                    color = Color(0xFF0F172A).copy(alpha = 0.6f),
                    shape = RoundedCornerShape(6.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(
                        text = "\"${item.example}\"",
                        fontSize = 13.sp,
                        fontStyle = FontStyle.Italic,
                        color = Color(0xFF94A3B8),
                        modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp)
                    )
                }
            }
        }
    }
}
