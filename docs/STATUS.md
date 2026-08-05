# Status do Projeto — Bounding Box Cardíaco (Cap. 2)

Registro de decisões e próximos passos por reunião de acompanhamento. Complementa
[step6v2_aorta_anchor.md](step6v2_aorta_anchor.md) (narrativa técnica completa, todos os números)
— este arquivo é o resumo de estado/decisão, não repete os detalhes de cálculo.

## Estado atual (reunião de 2026-07-29, Dany + Hélder)

### Decisão: testar em treino real duas opções de operação

Ambas usam o mesmo **Critério C** (`aorta_voxels`, `exclusion_pct=5%`) para exclusão de casos —
diferem só na tolerância a perda parcial de label coronário dentro do bbox.

- **Opção 1 — perda parcial tolerada**: `offset_percentile=95%` sobre a curva de perda real
  (`task5_percentile_bbox_tradeoff_real_vs_proxy.csv`, arm "com filtro critério C 5%"). `Rv`
  médio ≈ 46,9-47%. Não garante 100% dos voxels do label dentro do bbox para todo caso — mediana
  de perda real é 0%, mas há uma cauda de casos com perda severa já caracterizada (ver
  `docs/step6v2_aorta_anchor.md`, seção "Fechamento da tarefa 5").
- **Opção 2 — perda zero garantida**: `offset_percentile=98%` sobre a curva de N×Rv com perda
  real zero garantida (`task_n_rv_curve_zero_loss.csv`). `N_final`=848/1000 (84,8% do dataset
  retido), `Rv` médio=36,26%. Todo caso incluído tem 100% do label coronário dentro do bbox, por
  construção — ao custo de excluir ~15,2% do dataset original.

### Plano de validação combinado com o Dany

Não decidir entre as duas opções só pela curva de `Rv`/N — rodar as duas em treino real e
comparar a métrica de segmentação final (DSC/HD/AHD), além de VRAM/tempo de treino, para ver qual
estratégia de corte produz resultado mais consistente **depois do treino**, não só no
pré-processamento isolado.

### Item conceitual para a seção de limitações/discussão do Capítulo 2

Apontado pelo Dany, citando discussão com Alan: qualquer critério estatístico-populacional
(volume da aorta, número de círculos, tamanho do envelope) captura variabilidade normal da
população, mas não pode, por natureza, capturar anomalias anatômicas categóricas verdadeiras
(ex.: situs inversus, coração deslocado) — **não é uma limitação de volume de dados, é
categórica**: um padrão que não se repete na população não pode ser aprendido ou regularizado por
nenhuma estratégia baseada em comportamento médio. Identificar esses casos exigiria revisão por
radiologista, fora do escopo do pipeline determinístico. Registrar como **limitação explícita do
método**, não como falha.

### Pendências operacionais

- Confirmar acesso à máquina de treino com o Duca; verificar status do servidor "Curupira" (GPU)
  antes de iniciar o treino real.
- Risco recorrente de infraestrutura (terceira interrupção por energia/nobreak nesta
  investigação) — considerar no planejamento de tempo do treino real, mais longo e mais sensível
  a interrupção que os testes de calibração já feitos.
- Próximo check-in com o Dany: semana que vem, com resultado do treino nas duas opções.
