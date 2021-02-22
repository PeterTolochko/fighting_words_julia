using TextAnalysis


function preprocess_text(text)
	text = replace(text, r"[^ a-zA-Z0-9]" => "")
	text = lowercase(text)
	return text
end

function fighting_words(text₁, text₂; prior=.01, preprocess=false, comparison=true)
	"""Takes two arrays, each for a separate text category
	Outputs Ζ̂ scores (logged odds with Dirichlet prior)"""
	if preprocess
		text₁ = [preprocess_text(text) for text in text₁]
		text₂ = [preprocess_text(text) for text in text₂]
	end

    output_array = vcat(text₁, text₂)
	crps = Corpus([TokenDocument(x) for x in output_array])
	update_lexicon!(crps)
	update_inverse_index!(crps)

	remove_frequent_terms!(crps)
	update_lexicon!(crps)

	# doesn't always work for some reason
	# remove_sparse_terms!(crps)

	update_lexicon!(crps)
	update_inverse_index!(crps)

	terms = m.terms

	m = DocumentTermMatrix(crps, lexicon(crps))
	my_dtm = dtm(m, :dense)

	vocab_size = length(terms)

	# If using flat priors
	priors = [prior for i in 1:vocab_size]

	z_scores = zeros(vocab_size)

	count_matrix = zeros(2, vocab_size)

	count_matrix[1, :] = sum(my_dtm[1:length(l1), :], dims=1)
	count_matrix[2, :] = sum(my_dtm[length(l1):end, :], dims=1)

	a₀ = sum(priors)


	# for future
	# count_matrix[1:end .!=1, i]
	if comparison
		println("\nComparing two types; Obtaining ΔΖ̂...\n") 

		n₁ = sum(count_matrix[1, :])
		n₂ = sum(count_matrix[2, :])


		for i ∈ 1:vocab_size

			y_i = count_matrix[1, i]
			y_j = count_matrix[2, i]

			term₁ = log((y_i + priors[i]) / (n₁ + a₀ - y_i - priors[i]))
			term₂ = log((y_j + priors[i]) / (n₂ + a₀ - y_j - priors[i]))

			δ̂ = term₁ - term₂

			# compute variance
			σ² = 1 / (y_i + priors[i]) + 1 / (y_j + priors[i])

			z_scores[i] = δ̂ / sqrt(σ²)
		end
	else
		println("\nObtaining Ζ̂...\n")

		n₁ = sum(count_matrix[1, :])
		n₀ = sum(count_matrix, dims=(1, 2))[1] # total words in the sample


		for i ∈ 1:vocab_size

			y_i = count_matrix[1, i]
			y_j = y_i + count_matrix[2, i]

			term₁ = log((y_i + priors[i]) / (n₁ + a₀ - y_i - priors[i]))
			term₂ = log((y_j + priors[i]) / (n₀ + a₀ - y_j - priors[i]))

			δ̂ = term₁ - term₂

			# compute variance
			σ² = 1 / (y_i + priors[i]) + 1 / (y_j + priors[i])

			z_scores[i] = δ̂ / sqrt(σ²)

		end
	end
	terms = m.terms
	sorted_indices = sortperm(z_scores)

	return_list = []
	for i ∈ sorted_indices
		push!(return_list, (terms[i], z_scores[i]))
	end

	return return_list

end


fighting_words(output_array_1, output_array_2, prior=0.1, comparison=false)