using TextAnalysis

function preprocess_text(text)
	text = replace(text, r"[^ a-zA-Z0-9]" => "")
	text = lowercase(text)
	return text
end

function construct_dtm(text₁, text₂)
	output_array = vcat(text₁, text₂)
	crps = Corpus([TokenDocument(x) for x in output_array])
	# update_lexicon!(crps)
	update_inverse_index!(crps)
	remove_frequent_terms!(crps)
	# update_lexicon!(crps)
	# doesn't always work for some reason
	# remove_sparse_terms!(crps)
	update_lexicon!(crps)
	# update_inverse_index!(crps)
	m = DocumentTermMatrix(crps, lexicon(crps))
	my_dtm = dtm(m, :dense)
	return my_dtm, m
end

function fighting_words(text₁, text₂; α=.01, preprocess=false, comparison=true)
	"""Takes two arrays, each for a separate text category
	Outputs ζ̂ scores (for logged odds with Dirichlet prior)"""
	if preprocess
		text₁ = [preprocess_text(text) for text in text₁]
		text₂ = [preprocess_text(text) for text in text₂]
	end

	my_dtm, m = construct_dtm(text₁, text₂)

	vocab_size = length(m.terms)

	# If using flat priors
	if typeof(α) == Float64 || typeof(α) == Int64
		priors = [α for i ∈ 1:vocab_size]
	else
		priors = α
	end

	ζ̂_scores = zeros(vocab_size)

	count_matrix = zeros(2, vocab_size)

	count_matrix[1, :] = sum(my_dtm[1:length(l1), :], dims=1)
	count_matrix[2, :] = sum(my_dtm[length(l1):end, :], dims=1)

	α₀ = sum(priors)


	# for future
	# count_matrix[1:end .!=1, i]
	if comparison
		println("\nComparing two types; Obtaining Δζ̂...\n") 

		n₁ = sum(count_matrix[1, :])
		n₂ = sum(count_matrix[2, :])


		for i ∈ 1:vocab_size

			y_i = count_matrix[1, i]
			y_j = count_matrix[2, i]

			term₁ = log((y_i + priors[i]) / (n₁ + α₀ - y_i - priors[i]))
			term₂ = log((y_j + priors[i]) / (n₂ + α₀ - y_j - priors[i]))

			δ̂ = term₁ - term₂

			# compute variance
			σ² = 1 / (y_i + priors[i]) + 1 / (y_j + priors[i])

			ζ̂_scores[i] = δ̂ / sqrt(σ²)
		end
	else
		println("\nObtaining ζ̂...\n")

		n₁ = sum(count_matrix[1, :])
		n₀ = sum(count_matrix, dims=(1, 2))[1] # total words in the sample


		for i ∈ 1:vocab_size

			y_i = count_matrix[1, i]
			y_j = y_i + count_matrix[2, i]

			term₁ = log((y_i + priors[i]) / (n₁ + α₀ - y_i - priors[i]))
			term₂ = log((y_j + priors[i]) / (n₀ + α₀ - y_j - priors[i]))

			δ̂ = term₁ - term₂

			# compute variance
			σ² = 1 / (y_i + priors[i]) + 1 / (y_j + priors[i])

			ζ̂_scores[i] = δ̂ / sqrt(σ²)

		end
	end

	sorted_indices = sortperm(ζ_scores)

	return_list = Array{Any, 1}(undef, vocab_size)

	for i ∈ sorted_indices
		return_list[i] = (m.terms[i], ζ̂_scores[i])
	end

	return return_list

end


@benchmark fighting_words(output_array_1, output_array_2; α=0.1, comparison=true)


# BenchmarkTools.Trial: 
#   memory estimate:  700.85 MiB
#   allocs estimate:  8012542
#   --------------
#   minimum time:     1.235 s (13.69% GC)
#   median time:      1.280 s (13.04% GC)
#   mean time:        1.352 s (15.72% GC)
#   maximum time:     1.612 s (24.63% GC)
#   --------------
#   samples:          4
#   evals/sample:     1
