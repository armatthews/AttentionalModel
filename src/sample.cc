    map<string, int> translations;
    for (unsigned j = 0; j < 1000; ++j) {
      vector<WordId> target = attentional_model.SampleTranslation(source, ktSOS, ktEOS, 10);
      vector<string> words(target.size());
      for  (unsigned i = 0; i < target.size(); ++i) {
        words[i] = target_vocab.Convert(target[i]);
      }
      string translation = boost::algorithm::join(words, " ");
      translations[translation]++;

      if (ctrlc_pressed) {
        break;
      }
    }

    vector<pair<int, string> > translations2;
    for (auto it = translations.begin(); it != translations.end(); ++it) {
      translations2.push_back(make_pair(it->second, it->first));
    }

    auto comp = [](const pair<int, string>& a, const pair<int, string>& b) { return a.first > b.first || (a.first == b.first && a.second < b.second);};
    sort(translations2.begin(), translations2.end(), comp);

    for (auto it = translations2.begin(); it != translations2.end(); ++it) {
      cout << it->first << "\t" << it->second << endl;
    }
