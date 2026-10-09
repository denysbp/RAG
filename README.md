1 -fazer chunker:
    A. Markdown e texto: divisão hierárquica recursiva

    Tenta dividir por secções Markdown (#, ##, etc.).
    Se uma secção for demasiado grande, tenta dividi-la por parágrafos.
    Se um parágrafo continuar demasiado grande, divide por frases ou linhas.
    Se ainda for necessário, divide por caracteres para garantir que nunca ultrapassa o limite.

    B. Python: divisão consciente da estrutura do código

    Identifica funções e classes.
    Tenta manter cada função ou classe no mesmo chunk.
    Se uma função ou classe exceder o limite, divide-a por linhas ou blocos menores.
    Se um bloco individual for demasiado grande, divide-o por caracteres como último recurso.

    links:
        https://www.youtube.com/watch?v=8OJC21T2SL4
        https://reference.langchain.com/python/langchain-text-splitters/langchain_text_splitters
            DOCUMENTOS vLLM
                |
        Identificar extensão
            /             \
        .py                .md
        |                  |
        AST        LangChain Text Splitter
        |                  |
    Identificar funções   Dividir por secções
    e classes           e parágrafos
        |                  |
    Agrupar e dividir     Validar os limites
        |                  |
        +--------+---------+
                |
        Validar chunks
                |
        Guardar metadados
        - Caminho original
        - Índice inicial
        - Índice final
                |
        data/processed/
    
    


Fase 3 — Implementar a indexação

Cria um indexador que:

Percorra data/raw/.
Identifique o tipo de cada ficheiro.
Aplique a estratégia de divisão apropriada.
Guarde o texto dos excertos.
Registe o caminho exato de cada ficheiro.
Registe os intervalos de caracteres.
Construa o índice lexical.
Guarde os dados em data/processed/.

Não te esqueças de respeitar o limite de 2000 caracteres.

Fase 4 — Implementar a pesquisa

Cria um recuperador que receba uma pergunta e devolva os excertos mais relevantes.

Começa por implementar um método lexical, como TF-IDF ou BM25.

Depois, verifica se os caminhos dos ficheiros e os intervalos de caracteres estão corretos.

Não basta encontrar o texto certo: tens de devolver a localização exata desse texto no corpus.

Fase 5 — Implementar a geração de respostas

Utiliza os resultados da pesquisa para construir o contexto que será enviado ao Qwen.

O modelo deve responder à pergunta com base nos excertos recuperados.

A resposta deve ser organizada de acordo com os modelos Pydantic e exportada em JSON.