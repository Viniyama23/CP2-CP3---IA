"""Gera o dataset de suporte de TI (helpdesk) em PT-BR.
Treino: 2 formas de perguntar cada problema. Teste: uma 3ª forma, nunca vista no treino."""
import json, random, itertools
random.seed(42)

SYSTEM = "Você é um assistente de suporte de TI (helpdesk nível 1). Responda em português, em passos numerados, curtos e objetivos."

# (tema, [3 formas de perguntar], resposta)
SEEDS = [
 ("sem_internet", ["Estou sem internet, o que faço?", "Meu computador não conecta na rede. Como resolver?", "A internet caiu aqui, por onde começo a checar?"],
  "1. Verifique se o cabo/Wi-Fi está conectado.\n2. Reinicie o roteador (desligue 30s e ligue).\n3. Rode `ipconfig /release` e `ipconfig /renew` no CMD.\n4. Teste `ping 8.8.8.8`: se responder, o problema é DNS; use 8.8.8.8 como DNS.\n5. Se persistir, abra chamado informando o resultado do ping."),
 ("senha_esquecida", ["Esqueci minha senha de rede, como recupero?", "Não consigo logar com minha senha do Windows. O que fazer?", "Minha conta bloqueou depois de errar a senha, e agora?"],
  "1. Confirme se Caps Lock está desligado e o idioma do teclado correto.\n2. Use a opção de autoatendimento 'Esqueci minha senha', se disponível.\n3. Aguarde 15 a 30 minutos caso a conta esteja bloqueada por tentativas.\n4. Se continuar, solicite o reset ao suporte com seu nome e matrícula.\n5. No primeiro acesso, defina uma senha nova com 12+ caracteres."),
 ("pc_lento", ["Meu computador está muito lento.", "O PC trava toda hora, como melhorar?", "O Windows está demorando para abrir os programas, o que verifico?"],
  "1. Abra o Gerenciador de Tarefas (Ctrl+Shift+Esc) e veja quem usa mais CPU/memória.\n2. Desative programas desnecessários na aba 'Inicializar'.\n3. Libere espaço em disco (mínimo 15% livre) com a Limpeza de Disco.\n4. Instale atualizações pendentes e reinicie.\n5. Se o disco estiver em 100%, avalie trocar HDD por SSD."),
 ("impressora", ["A impressora não imprime, ajuda!", "Meu documento fica preso na fila de impressão.", "Impressora aparece offline no computador, como resolver?"],
  "1. Verifique energia, cabo/rede e se há papel ou atolamento.\n2. Em 'Impressoras e scanners', remova os trabalhos da fila.\n3. Reinicie o serviço 'Spooler de Impressão' (services.msc).\n4. Desmarque 'Usar impressora offline' nas opções da fila.\n5. Reinstale o driver do fabricante se o erro continuar."),
 ("vpn", ["A VPN não conecta, o que faço?", "Não consigo acessar a rede da empresa de casa.", "Minha VPN cai a cada poucos minutos, como corrigir?"],
  "1. Confirme que sua internet está funcionando sem a VPN.\n2. Feche e abra o cliente da VPN e tente de novo.\n3. Verifique se usuário, senha e MFA estão corretos.\n4. Desative temporariamente outras VPNs/proxies.\n5. Atualize o cliente da VPN; se falhar, envie o log ao suporte."),
 ("email_cheio", ["Minha caixa de e-mail está cheia.", "Não consigo enviar e-mails, aparece cota excedida.", "Como libero espaço no Outlook?"],
  "1. Esvazie as pastas 'Itens Excluídos' e 'Lixo Eletrônico'.\n2. Ordene por tamanho e remova e-mails com anexos grandes.\n3. Salve anexos importantes no OneDrive/rede.\n4. Use o Arquivamento automático do Outlook.\n5. Se necessário, peça aumento de cota ao suporte com justificativa."),
 ("tela_azul", ["Apareceu tela azul no meu computador.", "O PC reinicia sozinho com um erro azul, o que é?", "Como investigar uma tela azul (BSOD)?"],
  "1. Anote o código de parada exibido (ex.: MEMORY_MANAGEMENT).\n2. Reinicie e verifique se o erro se repete.\n3. Atualize drivers (principalmente vídeo e rede) e o Windows.\n4. Rode `sfc /scannow` e `chkdsk /f` como administrador.\n5. Teste a memória com o Diagnóstico de Memória do Windows e abra chamado com o código."),
 ("wifi_fraco", ["O Wi-Fi está muito fraco na minha sala.", "A conexão Wi-Fi oscila, o que fazer?", "Como melhorar o sinal de Wi-Fi no escritório?"],
  "1. Aproxime-se do roteador e compare a velocidade.\n2. Prefira a rede 5 GHz quando estiver perto.\n3. Mude o canal do roteador para um menos congestionado.\n4. Afaste o roteador de paredes, micro-ondas e metais.\n5. Se o problema for geral, solicite um repetidor ou ponto de acesso adicional."),
 ("phishing", ["Recebi um e-mail suspeito pedindo minha senha.", "Como saber se um e-mail é phishing?", "Cliquei em um link estranho, e agora?"],
  "1. Não responda e não abra anexos.\n2. Confira o remetente real e o destino do link (passe o mouse sem clicar).\n3. Se clicou, desconecte da rede e troque sua senha imediatamente.\n4. Use o botão 'Reportar phishing' ou encaminhe ao time de segurança.\n5. Ative MFA em todas as contas importantes."),
 ("backup", ["Como faço backup dos meus arquivos?", "Perdi um arquivo importante, dá para recuperar?", "Qual a melhor forma de proteger meus documentos contra perda?"],
  "1. Siga a regra 3-2-1: 3 cópias, 2 mídias, 1 fora do local.\n2. Salve documentos de trabalho no OneDrive/servidor, não só no desktop.\n3. Para recuperar, clique com o botão direito na pasta > 'Restaurar versões anteriores'.\n4. Confira a lixeira antes de qualquer outra ação.\n5. Teste a restauração periodicamente."),
 ("disco_cheio", ["Meu disco C: está cheio.", "Aparece pouco espaço em disco, como liberar?", "Onde está ocupando tanto espaço no meu PC?"],
  "1. Abra Configurações > Sistema > Armazenamento para ver o que ocupa espaço.\n2. Rode a Limpeza de Disco, incluindo arquivos temporários.\n3. Desinstale programas que não usa.\n4. Esvazie a pasta Downloads e a lixeira.\n5. Mova arquivos grandes para um disco externo ou para a nuvem."),
 ("monitor", ["Meu monitor está sem imagem.", "A tela fica preta mas o PC liga, o que é?", "O segundo monitor não é detectado, como resolver?"],
  "1. Confira cabos (HDMI/DP) e a entrada correta selecionada no monitor.\n2. Teste outro cabo ou outra porta de vídeo.\n3. Pressione Win+P e escolha 'Estender' ou 'Duplicar'.\n4. Atualize ou reinstale o driver de vídeo.\n5. Teste o monitor em outro computador para isolar a falha."),
 ("lentidao_sistema", ["O sistema da empresa está lento só para mim.", "O sistema interno demora para abrir, o que verifico?", "Outras pessoas acessam normal, mas o meu sistema trava. Por quê?"],
  "1. Teste em outro navegador ou em janela anônima.\n2. Limpe cache e cookies do navegador.\n3. Desative extensões e reinicie o navegador.\n4. Compare com um colega na mesma rede para isolar o problema.\n5. Abra chamado informando horário, mensagem de erro e print."),
]
PREFIXOS = ["", "Bom dia! ", "Urgente: ", "Estou no Windows 11. ", "Sou novo na empresa e não sei o que fazer. "]

def row(q, a):
    return {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}, {"role": "assistant", "content": a}]}

train, test = [], []
for _, qs, a in SEEDS:
    for pre, q in itertools.product(PREFIXOS, qs[:2]):
        train.append(row(pre + q, a))
    for pre in random.sample(PREFIXOS, 2):
        test.append(row(pre + qs[2], a))
random.shuffle(train)
for name, rows in (("train", train), ("test", test)):
    with open(f"data/{name}.jsonl", "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"train={len(train)} test={len(test)}")
